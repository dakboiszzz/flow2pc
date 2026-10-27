"""Synthetic contract/integration checks; never load the selected real teacher."""
from pathlib import Path
import csv
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from repc import cell_tracking as cell
from repc.single_cell import sha256_file, atomic_json
from scripts import prepare_cell_tracking as prepare
from scripts import train_cell_tracking as train
from scripts import run_cell_tracking_sweep as sweep


class CellContractTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(3)

    def test_teacher_uses_global_time_and_is_frozen(self):
        class Record(nn.Module):
            def __init__(self): super().__init__(); self.layer = nn.Linear(3, 2)
            def forward(self, x): self.input = x.detach().clone(); return self.layer(x)
        model = Record(); teacher = cell.CellTeacher(model)
        x = torch.randn(5, 2); teacher(.87, x)
        torch.testing.assert_close(model.input[:, :2], x)
        torch.testing.assert_close(model.input[:, 2], torch.full((5,), .87))
        self.assertFalse(model.training)
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        teacher(torch.linspace(0,1,5), x)
        torch.testing.assert_close(model.input[:, 2], torch.linspace(0,1,5))

    def test_endpoint_splits_and_border_ot_are_reproducible(self):
        clouds = [torch.arange(20,dtype=torch.float32).reshape(10,2)] * 115
        bundle = {"clouds":clouds, "train_source_rows":torch.arange(7),"train_end_rows":torch.arange(7),
                  "validation_source_rows":torch.arange(7,10),"validation_end_rows":torch.arange(7,10)}
        a,b = cell.endpoint_batch(bundle,32,torch.Generator().manual_seed(7))
        aa,bb = cell.endpoint_batch(bundle,32,torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(a,aa) and torch.equal(b,bb))
        self.assertLess(float(a.max()),14); self.assertLess(float(b.max()),14)
        a,b = cell.endpoint_batch(bundle,32,torch.Generator().manual_seed(7),validation=True)
        self.assertGreaterEqual(float(a.min()),14); self.assertGreaterEqual(float(b.min()),14)

    def test_validation_selection_cannot_hide_final_error_in_global_mean(self):
        class Zero(nn.Module):
            def velocity(self, x, t): return torch.zeros_like(x)
        student = Zero().train()
        validation = {name:{"values":torch.zeros(4,2),"times":torch.zeros(4),"target":torch.full((4,2),value)}
                      for name,value in (("global",1.),("late",2.),("final",3.))}
        result = cell.validation_metrics(student,{"velocity_validation":validation},"cpu")
        self.assertEqual(result["global_velocity_mse"],1.)
        self.assertEqual(result["final_velocity_mse"],9.)
        self.assertEqual(result["selection_score"],9.)
        self.assertTrue(student.training)

    def test_free_student_parameters_receive_velocity_gradients(self):
        student = cell.build_student(4,8,0)
        self.assertTrue(all(p.requires_grad for p in student.parameters()))
        self.assertFalse(student.fourier_zero_center)
        loss = (student.velocity(torch.randn(16,2),torch.rand(16))-1).square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters()))

    def test_clamped_log_scale_has_consistent_time_derivative(self):
        student = cell.build_student(4,4,0).double()
        with torch.no_grad(): student.log_scale_coefficients.fill_(30.)
        t = torch.tensor(.21,dtype=torch.double,requires_grad=True)
        _,_,logs,reported = student.leaf_parameters(t)
        actual = torch.autograd.grad(logs.sum(),t)[0]
        self.assertGreater(int((logs.abs()==20).sum()),0)
        torch.testing.assert_close(actual,reported.sum())

    def test_pixel_emd_and_paired_errors_keep_the_reference_scale(self):
        points = torch.zeros(4,2); path = points[None].expand(115,-1,-1).clone()
        bundle = {"teacher_path":path,"clouds":[points]*115,"times":torch.linspace(0,1,115),
                  "minimum":torch.tensor([10.,20.]),"scale":torch.tensor([2.,3.]),
                  "metric_source_rows":torch.arange(4),"metric_real_rows":{i:torch.arange(4) for i in cell.DISPLAY_FRAMES}}
        result = cell.score_paths(bundle,path+torch.tensor([1.,0.]))
        self.assertEqual(result["teacher_final_emd_pixels"],0.)
        self.assertAlmostEqual(result["student_final_emd_pixels"],2.)
        self.assertAlmostEqual(result["final_paired_rmse_pixels"],2.)
        self.assertEqual(len(result["scores"]),115)

    def test_synthetic_reference_training_overlays_and_resume(self):
        # A tiny synthetic constant-velocity checkpoint, not ali_cfm_model.pth.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); images=root/'images'; images.mkdir()
            masks=np.zeros((115,7,7),dtype=bool); masks[:,1:4,1:4]=True
            stack=root/'masks.npy';np.save(stack,masks)
            for i in cell.DISPLAY_FRAMES:Image.fromarray(np.zeros((7,7),dtype=np.uint8)).save(images/f't{i:03d}.tif')
            # Native definitions are imported only to create a miniature fixture.
            import sys,importlib
            sys.path.insert(0,str(cell.DEFAULT_ALI_ROOT))
            native=importlib.import_module('ali_cfm.nets')
            velocity=native.MLP(2,w=8,time_varying=True)
            generator=native.TrainableInterpolant(2,8,t_smooth=0.,time_varying=True)
            with torch.no_grad():
                for p in velocity.parameters():p.zero_()
                velocity.net[-1].bias.copy_(torch.tensor([.1,-.05]))
                for p in generator.parameters():p.zero_()
            checkpoint=root/'synthetic.pth'
            torch.save({"cfm_model":velocity.state_dict(),"interpolant":generator.state_dict(),
                "config":{"dim":2,"cfm_dim":8,"net_hidden":8,"normalize_dataset":True,"whiten":False,"cfm_ot":"border","t_smooth":0.},
                "min_max":[torch.ones(2),torch.full((2,),3.)]},checkpoint)
            digest=sha256_file(checkpoint);reference=root/'reference'
            common=['--checkpoint',str(checkpoint),'--expected-sha256',digest,'--device','cpu','--wandb-mode','disabled']
            prep_args=prepare.parse_args(common+['--mask-stack',str(stack),'--image-dir',str(images),'--output-dir',str(reference),
                '--metric-samples','4','--eval-batch-size','16'])
            prepare.execute(prep_args)
            self.assertEqual(json.loads((reference/'status.json').read_text())['status'],'complete')
            with patch.object(prepare,'load_teacher',side_effect=AssertionError('immutable reference should be reused')):
                prepare.execute(prep_args)
            bundle=torch.load(reference/'cache.pt',map_location='cpu',weights_only=False)
            self.assertFalse(set(bundle['train_source_rows'].tolist()) & set(bundle['validation_source_rows'].tolist()))
            self.assertEqual(bundle['teacher_path'].shape,(115,9,2))
            torch.testing.assert_close(bundle['teacher_path'][-1]-bundle['teacher_path'][0],torch.tensor([.1,-.05]).expand(9,2),atol=1e-5,rtol=1e-5)
            loaded_teacher,loaded_generator,*_=cell.load_teacher(cell.DEFAULT_ALI_ROOT,checkpoint,digest,'cpu')
            self.assertTrue(all(not p.requires_grad for p in loaded_teacher.parameters()))
            self.assertTrue(all(not p.requires_grad for p in loaded_generator.parameters()))
            output=root/'student'
            args=train.parse_args(common+['--reference-dir',str(reference),'--output-dir',str(output),
                '--num-latents','4','--time-embedding-dim','4','--batch-size','8','--steps','3',
                '--validate-every','1','--checkpoint-every','1','--log-every','1'])
            train.execute(args)
            for filename in ('student_overlay.png','comparison_overlay.png','metrics.json','predictions.pt','per_frame_metrics.csv'):
                self.assertTrue((output/'eval'/filename).is_file(),filename)
            metrics=json.loads((output/'eval/metrics.json').read_text())
            self.assertEqual(metrics['scores'][-1]['frame_number'],115)
            self.assertEqual(metrics['source_particles'],9)
            last=torch.load(output/'checkpoints/last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(last['scheduler']['last_epoch'],3)
            args.resume=True;train.execute(args)
            with (output/'metrics.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))),3)
            self.assertEqual(json.loads((output/'status.json').read_text())['status'],'complete')


class SchedulerTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.grid=self.root/'grid.json'
        atomic_json(self.grid,{"common":{"num_latents":4,"time_embedding_dim":4,"batch_size":8,"lr":.001,"steps":3,"seed":0,
            "validate_every":1,"checkpoint_every":1,"log_every":1},"runs":[{"name":f'student{i}'} for i in range(3)]})
        self.args=sweep.parse_args(['--grid',str(self.grid),'--output-root',str(self.root/'output'),'--wandb-mode','disabled'])
        with patch.object(sweep,'input_identity',return_value={'checkpoint_sha256':cell.EXPECTED_SHA256}):self.plan=sweep.build_plan(self.args)

    def test_shipped_grid_has_108_unique_named_students(self):
        args=sweep.parse_args([])
        with patch.object(sweep,'input_identity',return_value={}):plan=sweep.build_plan(args)
        self.assertEqual(len(plan['jobs']),108)
        self.assertEqual(len({n for n,_ in plan['jobs']}),108)
        for name,settings in plan['jobs']:
            self.assertEqual(settings['batch_size'],64);self.assertEqual(settings['steps'],50000)
            self.assertEqual(settings['wandb_project'],'flow2pc')
            self.assertTrue(all(x in name for x in ('h','f','lr','b64','s','steps50000')))
        command=sweep.task_command(args,('train','student',{},0),'cuda:1')
        self.assertIn('train_cell_tracking.py',command[2]);self.assertIn('--resume',command)

    def test_scheduler_uses_multiple_gpus_and_skips_complete_runs(self):
        created=[];owner=self
        class Process:
            def __init__(self,command,**kwargs):
                self.pid=800000+len(created);self.command=command;self.returncode=None;self.polls=0
                self.output=Path(command[command.index('--output-dir')+1]);self.device=command[command.index('--device')+1]
                self.prepare=command[2].endswith('prepare_cell_tracking.py');created.append(self)
                running=[p for p in created if p.returncode is None]
                owner.assertLessEqual(len(running),4)
                if not self.prepare:owner.assertTrue((owner.args.output_root/'reference/metrics.json').exists())
            def poll(self):
                if self.returncode is not None:return self.returncode
                self.polls+=1
                if self.polls<2:return None
                self.returncode=0
                if self.prepare:atomic_json(self.output/'metrics.json',{'fixture':True})
                else:
                    atomic_json(self.output/'status.json',{'status':'complete','best_selection_score':.01})
                    atomic_json(self.output/'eval/metrics.json',{'student_final_emd_pixels':1.})
                    for name in ('student_overlay.png','comparison_overlay.png'):(self.output/'eval'/name).write_text('fixture')
                return 0
            def wait(self,timeout=None):return self.returncode
        with patch.object(sweep.subprocess,'Popen',Process),patch.object(sweep,'gpu_free_memory',return_value={'cuda:0':30*2**30,'cuda:1':30*2**30}),patch.object(sweep.time,'sleep'):
            sweep.run(self.args,self.plan,['cuda:0','cuda:1'])
            self.assertEqual(len(created),4)
            self.assertEqual({p.device for p in created if not p.prepare},{'cuda:0','cuda:1'})
            sweep.run(self.args,self.plan,['cuda:0','cuda:1'])
            self.assertEqual(sum(not p.prepare for p in created),3)
            self.assertTrue((self.args.output_root/'selected_student.json').exists())

    def test_dry_run_does_not_launch_workers_or_load_checkpoint(self):
        self.args.dry_run=True
        with patch.object(sweep,'parse_args',return_value=self.args),patch.object(sweep,'build_plan',return_value=self.plan),\
             patch.object(sweep.torch.cuda,'device_count',return_value=2),patch.object(sweep.subprocess,'Popen') as launch,\
             patch.object(torch,'load',side_effect=AssertionError('checkpoint execution forbidden')),patch('builtins.print'):
            sweep.main()
        launch.assert_not_called();self.assertFalse(self.args.output_root.exists())

    def test_reference_failure_prevents_student_launch(self):
        created=[]
        class Failure:
            def __init__(self,command,**kwargs):
                self.pid=810000;self.command=command;created.append(self)
                kwargs['stdout'].write('Invalid teacher reference fixture\n');kwargs['stdout'].flush()
            def poll(self):return 1
            def wait(self,timeout=None):return 1
        with patch.object(sweep.subprocess,'Popen',Failure),patch.object(sweep,'gpu_free_memory',return_value={'cuda:0':30*2**30,'cuda:1':30*2**30}),\
             patch.object(sweep.time,'sleep'),patch.object(sweep.os,'killpg'):
            with self.assertRaises(SystemExit):sweep.run(self.args,self.plan,['cuda:0','cuda:1'])
        self.assertEqual(len(created),1)
        self.assertIn('prepare_cell_tracking.py',created[0].command[2])

    def test_oom_worker_is_retried_without_cancelling_other_students(self):
        created=[];attempts={}
        class Process:
            def __init__(self,command,**kwargs):
                self.pid=820000+len(created);self.command=command;self.returncode=None;self.polls=0
                self.output=Path(command[command.index('--output-dir')+1]);self.log=kwargs['stdout']
                self.prepare=command[2].endswith('prepare_cell_tracking.py')
                key=str(self.output);attempts[key]=attempts.get(key,0)+1;self.attempt=attempts[key];created.append(self)
            def poll(self):
                if self.returncode is not None:return self.returncode
                self.polls+=1
                if self.polls<2:return None
                if not self.prepare and self.output.name.startswith('student0-') and self.attempt==1:
                    self.log.write('CUDA error: out of memory\n');self.log.flush();self.returncode=1;return 1
                self.returncode=0
                if self.prepare:atomic_json(self.output/'metrics.json',{'fixture':True})
                else:
                    atomic_json(self.output/'status.json',{'status':'complete','best_selection_score':.01})
                    atomic_json(self.output/'eval/metrics.json',{'student_final_emd_pixels':1.})
                    for name in ('student_overlay.png','comparison_overlay.png'):(self.output/'eval'/name).write_text('fixture')
                return 0
            def wait(self,timeout=None):return self.returncode
        with patch.object(sweep.subprocess,'Popen',Process),patch.object(sweep,'gpu_free_memory',return_value={'cuda:0':30*2**30,'cuda:1':30*2**30}),\
             patch.object(sweep.time,'sleep'),patch.object(sweep.os,'killpg'):
            sweep.run(self.args,self.plan,['cuda:0','cuda:1'])
        self.assertEqual(sum(not p.prepare for p in created),4)
        self.assertEqual(json.loads((self.args.output_root/'sweep_status.json').read_text())['status'],'complete')


if __name__=='__main__':unittest.main()
