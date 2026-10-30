"""Native knot regression checks, prepared for explicit user execution only."""

import ast
import copy
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from flow2pc.pretrain.models.alicfm._source import knot_training
from flow2pc.pretrain.models.alicfm.profiles import make_models, train_knot
from flow2pc.pretrain.models.alicfm.trainer import ALICFM, MethodConfig

ROOT = Path(__file__).resolve().parents[1]
REVISION = "a35963ef226e1bbdc6344a944c50e0f42ed67335"


@pytest.fixture
def native_tree():
    upstream = ROOT.parent / "ali-cfm"
    if not (upstream / ".git").exists():
        pytest.skip("Supplied native knot source history is unavailable")
    source = subprocess.check_output(
        ["git", "-C", str(upstream), "show", f"{REVISION}:train_teacher.py"],
        text=True,
    )
    return ast.parse(source)


def config():
    return yaml.safe_load((ROOT / "configs/pretrain/alicfm_knot.yaml").read_text())


def test_numpy_curve_sampler_is_the_native_lifted_function(native_tree):
    native_train = next(
        n for n in native_tree.body if isinstance(n, ast.FunctionDef) and n.name == "train"
    )
    sampler = next(
        n for n in native_train.body if isinstance(n, ast.FunctionDef)
        and n.name == "get_knot_data"
    )
    migrated = ast.parse(Path(knot_training.__file__).read_text())
    migrated_sampler = next(
        n for n in migrated.body if isinstance(n, ast.FunctionDef)
        and n.name == "get_knot_data"
    )
    assert ast.dump(sampler) == ast.dump(migrated_sampler)


def native_reference(native_tree):
    """Extract original model setup and both complete loops, without exports.

    Only the device, batch size and update budgets become harness arguments.
    Native sampler, optimizer setup and all training statements stay intact.
    """
    native_tree = copy.deepcopy(native_tree)
    train = next(
        n for n in native_tree.body if isinstance(n, ast.FunctionDef) and n.name == "train"
    )
    arguments = ast.parse(
        "def run_reference(device, batch_size, n_epochs_interpolant, n_epochs_flow): pass"
    ).body[0].args
    body = []
    for node in train.body:
        targets = (
            {n.id for n in node.targets if isinstance(n, ast.Name)}
            if isinstance(node, ast.Assign) else set()
        )
        if "teacher_checkpoint" in targets:
            break
        if targets & {"device", "batch_size", "n_epochs_interpolant", "n_epochs_flow"}:
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "print"
        ):
            continue
        body.append(node)
    body.extend(ast.parse(
        'return {"states": [m.state_dict() for m in '
        '(generator_model, discriminator, velocity_model)], '
        '"optimizers": [o.state_dict() for o in (opt_G, opt_D, opt_V)], '
        '"losses": [d_loss.item(), g_loss.item(), reg_loss.item(), v_loss.item()]}'
    ).body)
    harness = ast.FunctionDef(
        name="run_reference", args=arguments, body=body, decorator_list=[]
    )
    definitions = [
        n for n in native_tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef))
        or isinstance(n, ast.FunctionDef) and n.name == "get_dx_dt"
    ]
    module = ast.fix_missing_locations(
        ast.Module(body=[*definitions, harness], type_ignores=[])
    )
    namespace = {}
    exec(compile(module, "<native-knot-reference>", "exec"), namespace)
    return namespace["run_reference"]


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_equal(a, b)
    else:
        assert left == right


def test_full_native_and_migrated_updates_and_rng_match(native_tree):
    reference = native_reference(native_tree)
    torch.manual_seed(19)
    np.random.seed(23)
    expected = reference("cpu", 4, 2, 2)
    expected_numpy, expected_torch = np.random.get_state(), torch.get_rng_state()

    cfg = config()
    cfg["training"].update(
        device="cpu", batch_size=4, num_ali_train_steps=2, num_cfm_train_steps=2
    )
    ALICFM().validate(cfg)
    torch.manual_seed(19)
    np.random.seed(23)
    generator, discriminator, field = make_models(dict(cfg["model"]["params"], dim=2), "cpu")
    settings = MethodConfig(**cfg["model"]["params"], **cfg["training"])
    optimizers = {
        "generator": torch.optim.Adam(generator.parameters(), lr=settings.interpolant_gen_lr),
        "discriminator": torch.optim.Adam(discriminator.parameters(), lr=settings.interpolant_disc_lr),
        "cfm": torch.optim.Adam(field.parameters(), lr=settings.cfm_lr),
    }
    rows = []
    dataset = types.SimpleNamespace(
        heldout=None, params=cfg["data"]["params"],
        preprocessing={"normalise": False, "whiten": False},
        metadata=types.SimpleNamespace(model_times=tuple(range(120))),
    )
    train_knot(
        generator, discriminator, field, optimizers, dataset, settings,
        types.SimpleNamespace(log=rows.append),
    )
    assert_equal(expected["states"], [m.state_dict() for m in (generator, discriminator, field)])
    assert_equal(expected["optimizers"], [o.state_dict() for o in optimizers.values()])
    assert_equal(expected["losses"], [
        rows[1]["ali/gan/d_loss"], rows[1]["ali/gan/g_loss"],
        rows[1]["ali/gan/regularizer"], rows[-1]["ali/cfm/loss"],
    ])
    actual_numpy = np.random.get_state()
    assert expected_numpy[0] == actual_numpy[0]
    np.testing.assert_array_equal(expected_numpy[1], actual_numpy[1])
    assert expected_numpy[2:] == actual_numpy[2:]
    assert_equal(expected_torch, torch.get_rng_state())
    # Preserve both the native (x,t) call and the frozen adapter's joined input.
    x, t = torch.zeros(2, 2), torch.tensor([[0.25], [0.75]])
    assert_equal(field(x, t), field(torch.cat([x, t], dim=-1)))


@pytest.mark.parametrize("key,value", [
    ("normalise", True), ("whiten", True), ("heldout_index", 1),
])
def test_native_knot_rejects_silent_data_protocol_changes(key, value):
    cfg = config()
    cfg["data"]["params"][key] = value
    with pytest.raises(ValueError, match="native knot trainer"):
        ALICFM().validate(cfg)
