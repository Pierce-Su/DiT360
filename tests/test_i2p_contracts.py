"""CPU-only real torch/API contracts with tiny mocks, NOT checkpoint execution."""
import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
try:
    import torch
except ImportError:
    torch = None
import i2p


@unittest.skipIf(torch is None, 'requires isolated torch environment')
class PipelineContracts(unittest.TestCase):
    def test_encode_generator_reaches_vae(self):
        from pa_src.pipeline import RFPanoInversionParallelFluxPipeline as P
        seen = []
        class Dist:
            def sample(self, generator=None):
                seen.append(generator)
                return torch.randn((1,2,4), generator=generator)
        cfg = type('Config', (dict,), {'__getattr__':dict.__getitem__})(sample_size=1024,shift_factor=0.,scaling_factor=1.)
        obj = NS(vae=NS(config=cfg, encode=lambda x:NS(latent_dist=Dist())), device='cpu',
                 image_processor=NS(preprocess=lambda **kw:torch.zeros((1,3,16,32)),postprocess=lambda **kw:[]))
        a = torch.Generator().manual_seed(42)
        b = torch.Generator().manual_seed(42)
        x, _ = P.encode_image(obj, None, dtype=torch.float32, generator=a)
        y, _ = P.encode_image(obj, None, dtype=torch.float32, generator=b)
        self.assertIs(seen[0],a)
        self.assertTrue(torch.equal(x,y))

    def test_invert_vae_and_noise_same_stream(self):
        from pa_src.pipeline import RFPanoInversionParallelFluxPipeline as P
        import pa_src.pipeline as mod
        seen=[]
        def encode(image, **kw):
            seen.append(kw['generator'])
            return torch.randn((1,2,4), generator=kw['generator']), []
        class Progress:
            def __enter__(self): return self
            def __exit__(self,*a): pass
            def update(self): pass
        cfg=NS(base_image_seq_len=256,max_image_seq_len=4096,base_shift=.5,max_shift=1.16)
        obj=NS(text_encoder=NS(dtype=torch.float32),transformer=NS(config=NS(in_channels=16,guidance_embeds=False)),
               _execution_device='cpu', encode_image=encode, vae_scale_factor=8,
               prepare_latents_inversion=lambda *a:(a[-1],torch.zeros((2,3))),scheduler=NS(config=cfg),
               get_timesteps=lambda *a:(torch.tensor([1.,.5]),torch.tensor([1.,.5,0.]),2),
               encode_prompt=lambda **kw:(None,None,None), progress_bar=lambda **kw:Progress(),joint_attention_kwargs=None)
        obj.transformer=lambda **kw:(torch.zeros_like(kw['hidden_states']),)
        obj.transformer.config=NS(in_channels=16,guidance_embeds=False)
        original=torch.randn
        def tracked(*args,**kwargs):
            seen.append(kwargs.get('generator'))
            return original(*args,**kwargs)
        rng=torch.Generator().manual_seed(42)
        with patch.object(mod,'retrieve_timesteps',return_value=(torch.tensor([1.,.5]),2)), patch.object(torch,'randn',side_effect=tracked):
            inv,image,ids=P.invert(obj,None,height=16,width=32,num_inversion_steps=2,generator=rng)
        self.assertTrue(all(g is rng for g in seen))
        self.assertEqual(tuple(inv.shape),(1,4,4))
        self.assertEqual(tuple(ids.shape),(4,3))

    def test_runner_streams_params_and_two_sample_lifecycle(self):
        events=[]
        class Transformer:
            attn_processors={'native':object()}
            def set_attn_processor(self,procs):
                self.attn_processors=dict(procs)
                events.append('restore' if 'native' in procs else 'personalized')
        class Pipe:
            vae_scale_factor=8
            _execution_device='cpu'
            transformer=Transformer()
            def invert(self,**kw):
                self.inv_rng=kw['generator']
                self.inv_value=torch.randn(4,generator=self.inv_rng)
                assert 'native' in self.transformer.attn_processors
                events.append('invert')
                return torch.zeros((1,8320,4)),torch.zeros((1,8320,4)),torch.zeros((8320,3))
            def __call__(self,prompts,**kw):
                assert kw['generator'] is not self.inv_rng
                self.gen_value=torch.randn(4,generator=kw['generator'])
                assert kw['guidance_scale']==3.5 and kw['t_l']==.5 and kw['max_sequence_length']==512
                assert prompts==['p','p']
                events.append('generate')
                return NS(images=[None,'result'])
        def install(t,*args): t.set_attn_processor({'personalized':object()})
        p=Pipe(); values=[]
        for _ in range(2):
            result=i2p.run_pipeline(p,np.zeros((1024,2048,3),np.uint8),np.zeros((1024,2048),bool),'p',42,torch,install)
            self.assertEqual(result,'result')
            values.append((p.inv_value,p.gen_value))
        self.assertEqual(events,['restore','invert','personalized','generate','restore']*2)
        self.assertTrue(torch.equal(values[0][0],values[1][0]))
        self.assertTrue(torch.equal(values[0][1],values[1][1]))

    def test_runner_restores_on_failure(self):
        class Transformer:
            attn_processors={'native':object()}
            def set_attn_processor(self,p): self.attn_processors=dict(p)
        def fail(**kw): raise RuntimeError('intentional inversion failure')
        p=NS(vae_scale_factor=8,_execution_device='cpu',transformer=Transformer(),invert=fail)
        with self.assertRaises(RuntimeError):
            i2p.run_pipeline(p,np.zeros((32,64,3),np.uint8),np.zeros((32,64),bool),'p',42,torch,None)
        self.assertIn('native',p.transformer.attn_processors)

    def test_real_processor_and_tiny_flux_forward_cpu(self):
        from diffusers import FluxTransformer2DModel
        from pa_src.attn_processor import PersonalizeAnythingAttnProcessor, set_flux_transformer_attn_processor
        model=FluxTransformer2DModel(in_channels=4,out_channels=4,num_layers=1,num_single_layers=1,
            attention_head_dim=8,num_attention_heads=2,joint_attention_dim=16,pooled_projection_dim=16,
            axes_dims_rope=(2,2,4),guidance_embeds=True)
        mask=torch.tensor([[1.],[0.],[1.],[0.]])
        set_flux_transformer_attn_processor(model,lambda name,dh,nh,ap:PersonalizeAnythingAttnProcessor(name,mask,'cpu',tau=.5,img_dims=4))
        for t in [.8,.2]:
            with torch.no_grad():
                result=model(hidden_states=torch.randn((2,4,4)),encoder_hidden_states=torch.randn((2,512,16)),
                    timestep=torch.ones(2)*t,guidance=torch.ones(2)*3.5,pooled_projections=torch.zeros((2,16)),
                    txt_ids=torch.zeros((512,3)),img_ids=torch.zeros((4,3)),joint_attention_kwargs={'timestep':t},return_dict=False)[0]
            self.assertEqual(tuple(result.shape),(2,4,4))
            self.assertTrue(torch.isfinite(result).all())

    def test_transfer_gate_not_rebound(self):
        from pa_src.pipeline import RFPanoInversionParallelFluxPipeline as P
        self.assertEqual(P.__call__.__wrapped__.__defaults__[-1],.5)
        tree=ast.parse(Path('pa_src/pipeline.py').read_text())
        assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='t_l' for x in n.targets)]
        self.assertEqual(assignments,[])


if __name__=='__main__': unittest.main()
