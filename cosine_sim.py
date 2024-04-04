import torch
import torch.nn as nn

def pull_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    
    return(feature_extractor)

def test_similarity(f_e, orig, gen, distractor_1, distractor_2):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    orig = orig[:,0:3,:,:]
    gen = gen[:,0:3,:,:]
    distractor_1 = distractor_1[:,0:3,:,:]
    distractor_2 = distractor_2[:,0:3,:,:]
    orig_output = torch.flatten(f_e(orig), start_dim = 1) 
    gen_output = torch.flatten(f_e(gen), start_dim = 1)
    distractor_1_output = torch.flatten(f_e(distractor_1), start_dim = 1)
    distractor_2_output = torch.flatten(f_e(distractor_2), start_dim = 1)

    orig_gen_cos = cos(orig_output, gen_output)
    distractor_1_gen_cos = cos(distractor_1_output, gen_output)
    distractor_2_gen_cos = cos(distractor_2_output, gen_output)
    diff_1 = torch.mean(orig_gen_cos) - torch.mean(distractor_1_gen_cos)
    diff_2 = torch.mean(orig_gen_cos) - torch.mean(distractor_2_gen_cos)
    if(diff_1 > diff_2):
        return(diff_2.detach().cpu().numpy())
    else:
        return(diff_1.detach().cpu().numpy())
if(__name__ == '__main__'):
    orig = torch.randn([10,4,23,137])
    gen = torch.randn([10,4,23,137])
    distractor_1 = torch.randn([10,4,23,137])
    distractor_2 = torch.randn([10,4,23,137])
    f_e = pull_model()
    test_similarity(f_e, orig, gen, distractor_1, distractor_2)