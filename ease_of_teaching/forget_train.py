from __future__ import print_function
from __future__ import division
import torch
import parser
import os
import random
import copy
import sys
import argparse
import wandb
import h5py

parser = argparse.ArgumentParser(description='Referential game settings')

parser.add_argument('--gpu', type=int, default=0, help='which gpu if we use gpu')
parser.add_argument('--fname', type=str, default='test', help='folder name to save results')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--jupyter', action='store_true') 
parser.add_argument('--slambda', type=float, default=0.1, help='speaker regularization hyperparameter')
parser.add_argument('--rlambda', type=float, default=0.1, help='listener regularization hyperparameter')
parser.add_argument('--receiverNum', type=int, default=1, help='number of listeners in the population')
parser.add_argument('--topk', type=int, default=3, help='number of top messages when we probe language')
parser.add_argument('--evaluateSize', type=int, default=1000, help='the batch size of test objects when not enumeration')

parser.add_argument('--trainIters', type=int, default=300000, help='number of training iterations')
parser.add_argument('--resetNum', type=int, default=50, help='number of reset iterations')
parser.add_argument('--batchSize', type=int, default=100)
parser.add_argument('--easy_limit', type=int, default=5, help='max number of easy distractors in easy sample')
parser.add_argument('--sLearnRate', type=float, default=0.001)
parser.add_argument('--rLearnRate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--reset_receiver', action='store_true', help='reset receiver completely') 
parser.add_argument('--freeze_receiver', action='store_true', help='freeze receiver') 
parser.add_argument('--freeze_after_gen_num', type=int, default=1, help='freeze receiver after n generations')

parser.add_argument('--forget_sender', action='store_true', help='partial reset of sender') 
parser.add_argument('--forget_receiver', action='store_true', help='partial reset of receiver') 
parser.add_argument('--same_mask', action='store_true', help='reset the same weights each generation (mask is predetermined)') 
parser.add_argument('--weight_mask', action='store_true', help='choose weights with the smallest magnitude to reset') 
parser.add_argument('--reset_to_zero', action='store_true', help='set reset weights to 0. if false, reinitialize') 
parser.add_argument('--sender_keep_perc', type=float, default=1, help='percentage of sender weights to keep')
parser.add_argument('--receiver_keep_perc', type=float, default=1, help='percentage of receiver weights to keep')
parser.add_argument("--group_vars", type=str, nargs='+', default="", help="variables used for grouping in wandb")
parser.add_argument(
    '--no_wandb',
    action='store_true',
    help='do not write to wandb')
parser.add_argument(
    '--track_examples',
    action='store_true',
    help='save example tracker')
parser.add_argument(
    '--save_weights',
    action='store_true',
    help='save weights')

args_raw = parser.parse_args()
if not args_raw.no_wandb:
    if len(args_raw.group_vars) > 0:
        if len(args_raw.group_vars) == 1:
            group_name = args_raw.group_vars[0] + str(getattr(args_raw, args_raw.group_vars[0]))
        else:
            group_name = args_raw.group_vars[0] + str(getattr(args_raw, args_raw.group_vars[0]))
            for var in args_raw.group_vars[1:]:
                group_name = group_name + '_' + var + str(getattr(args_raw, var))
        wandb.init(project="fortuitous_forgetting",
               group="ease_of_teaching")
        for var in args_raw.group_vars:
            wandb.config.update({var:getattr(args_raw, var)})
    else:
            wandb.init(project="fortuitous_forgetting",
                       group="ease_of_teaching")
            for var in args_raw.group_vars:
                wandb.config.update({var: getattr(args_raw, var)})
            
args = vars(args_raw) # convert python object to dict
# args = parser.parse()  # parsed argument from CLI
args['device'] = torch.device("cuda:" + str(args['gpu']) if torch.cuda.is_available() else "cpu")
# if not os.path.exists(args['fname']):
try:
    os.makedirs(args['fname'])    
except:
    print('folder already exists')    

# dataset hyperparameters
args['numColors'] = 8 
args['numShapes'] = 4 
args['attrSize'] = args['numColors'] + args['numShapes']  # colors + shapes

# game settings
args['vocabSize'] = 8
args['messageLen'] = 2
args['distractNum'] = 5  # including targets

# training hyperparameters
args['resetIter'] = args['trainIters'] // args['resetNum']  # life of a receiver: 6K
args['deterResetNums'] = 30
args['deterResetIter'] = 1000

# population of receivers training
args['population'] = False

# model hyperparameters
args['hiddenSize'] = 100 

print(args)

from dataGenerator import Dataset
import utility
import numpy as np
if args['population']:
    from popgame import popGuessGame
else:
    from game import GuessGame

torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])
np.random.seed(args['seed'])
random.seed(args['seed'])

torch.backends.cudnn.deterministic=True

# @title Train
team = GuessGame(args)
# get data
data = Dataset(args['numColors'], args['numShapes'], args['attrSize'])
train_np = data.getTrain()
util = utility.Utility(args, data)

sloss_l = np.zeros(args['trainIters'])
rloss_l = np.zeros(args['trainIters'])
trainAccuracy_l = np.zeros(args['trainIters'])
entropy_l = np.zeros(args['trainIters'])

# easy-to-teach evaluation
evalAcc_l = np.zeros((args['resetNum'] // 10, args['deterResetNums'], args['deterResetIter']))

dTopo = np.zeros(args['resetNum']+1)
dEntropy = np.zeros(args['resetNum']+1)

def convert_data_to_tensor(indices):
    color = np.zeros([32, 5, 8], dtype=np.float32)
    shape = np.zeros([32, 5, 4], dtype=np.float32)

    # fetch the batchid and turn color/shape index into one hot nunpy vertor
    for i in range(32):
        for j in range(5):
            colorindex = indices[i][j][0]
            shapeindex = indices[i][j][1]
            color[i][j][colorindex] = 1
            shape[i][j][shapeindex] = 1

    instances = np.concatenate([color, shape], axis=2)
    targets = instances[np.arange(32), 0, :]
    
    return instances, targets

if args_raw.track_examples:
    time_tracker=[]
    topo_tracker=[]
    message_tracker = []
    
    #generate single data points
    all_color_hard_examples = []
    for color, shape in train_np:
        instance = []
        other_colors = [i for i in range(8) if i != color]
        other_shapes = [i for i in range(4) if i != shape]
        instance.append([color, shape])
        for sh in other_shapes:
            instance.append([color, sh])
        instance.append([other_colors[0], shape])
        all_color_hard_examples.append(np.vstack(instance))
    all_color_hard_examples = np.stack(all_color_hard_examples)
    hard_color_inputs, hard_color_targets = convert_data_to_tensor(all_color_hard_examples)
    hard_color_tracker = []
    
    
    all_shape_hard_examples = []
    for color, shape in train_np:
        instance = []
        other_colors = [i for i in range(8) if i != color]
        other_shapes = [i for i in range(4) if i != shape]
        instance.append([color, shape])
        rand_sample = np.random.choice(len(other_colors), 4, False)
        for c in rand_sample:
            instance.append([other_colors[c], shape])
        all_shape_hard_examples.append(np.vstack(instance))
    all_shape_hard_examples = np.stack(all_shape_hard_examples)
    hard_shape_inputs, hard_shape_targets = convert_data_to_tensor(all_shape_hard_examples)
    hard_shape_tracker = []

    
    all_easy_examples = []
    for color, shape in train_np:
        instance = []
        other_colors = [i for i in range(8) if i != color]
        other_shapes = [i for i in range(4) if i != shape]
        instance.append([color, shape])
        k=0
        while k < 4:
            c = np.random.choice(len(other_colors),1)
            s = np.random.choice(len(other_shapes),1)
            color = other_colors[c[0]]
            shape = other_shapes[s[0]]
            pair = [color, shape]
            if pair not in instance:
                instance.append(pair)
                k += 1
        all_easy_examples.append(np.vstack(instance))
    all_easy_examples = np.stack(all_easy_examples)
    easy_inputs, easy_targets = convert_data_to_tensor(all_easy_examples)
    easy_tracker = []


    all_random_examples = []
    for color, shape in train_np:
        instance = []
        instance.append([color, shape])
        k=0
        while k < 4:
            c = np.random.choice(8,1)
            s = np.random.choice(4,1)
            color = c[0]
            shape = s[0]
            pair = [color, shape]
            if pair not in instance:
                instance.append(pair)
                k += 1
        all_random_examples.append(np.vstack(instance))
    all_random_examples = np.stack(all_random_examples)
    random_inputs, random_targets = convert_data_to_tensor(all_random_examples)
    random_tracker = []


def torch_bernoulli(p, size):
    return torch.rand(size) < p

def save_weight_one_iter(model):
    weight_all = []
    for name, param in model.named_parameters():
        weight_all.append(param.data.clone().cpu().numpy().flatten())

    weight_all = np.concatenate(weight_all)
    
    return weight_all

if args_raw.save_weights:
    sbot_weights_all_time = []
    rbot_weights_all_time = []
    time_ind = []
    sloss_tracker = []
    rloss_tracker = []
    acc_tracker = []
    topo_tracker = []


def eval_teach_speed(eval_ind, data, team):
    # make deterministic, just when training let sender speaks deterministic language
    print('Evaluate the teaching speed after reset for ' + str(10 * (eval_ind+1)) + ' evaluations')

    for i in range(args['deterResetNums']):
        # evaluate before reset
        print('Reset the ' + str(i+1) + 'th receiver with deterministic language')  # start from 1
        team.resetReceiver(sOpt=False)
        
        for j in range(args['deterResetIter']):
            candidates, targets = data.getBatchData(train_np, args['batchSize'], args['distractNum'])
            sloss, rloss, message, rewards, _, _, _ = team.forward(targets, candidates, evaluate=False, sOpt=True, rOpt=True,
                                                                stochastic=False)  # speak in evaluate mode
            team.backward(sloss, rloss, sOpt=False) 
            evalAcc_l[eval_ind][i][j] = rewards.sum().item() / args['batchSize'] * 100  # reward +1 0

            # print intermediate results during training
            if j == 0 or (j + 1) % 100 == 0:
                record = 'Iteration ' + str(i * args['deterResetIter'] + j + 1) \
                         + ' Training accuracy ' + str(np.round(evalAcc_l[eval_ind][i][j], decimals=2)) + '%\n'
                print(record)
                if not args_raw.no_wandb:
                    wandb.log({'Eval Speed Iter': i * args['deterResetIter'] + j + 1, 'Eval Speed Training accuracy': np.round(evalAcc_l[eval_ind][i][j], decimals=2)})

with torch.no_grad():
    dTopo[0], dEntropy[0], prevLangD = util.get_sender_language(team, neural=True)  # evaluate all group performance
    
if not args_raw.no_wandb:
    wandb.log({'Iter': 0, 'Topographic Similarity': dTopo[0]})
        
if args_raw.forget_sender and args_raw.same_mask:
    if args_raw.weight_mask:
        mask_dict = {}
        for name, param in team.sbot.named_parameters():
            weight_mag = torch.abs(param.detach().clone())
            topk = torch.topk(weight_mag.flatten(), k=int(weight_mag.nelement()*(1-args_raw.sender_keep_perc)), largest=False)
            temp_mask = torch.ones(weight_mag.nelement())
            temp_mask[topk.indices] = 0
            mask_dict[name] = temp_mask.bool().view(weight_mag.shape)
    else:
        mask_dict = {}
        for name, param in team.sbot.named_parameters():
            mask_dict[name] = torch_bernoulli(args_raw.sender_keep_perc, param.shape)
                
if args_raw.forget_receiver and args_raw.same_mask:
    if args_raw.weight_mask:
        mask_dict_rec = {}
        for name, param in team.rbot.named_parameters():
            weight_mag = torch.abs(param.detach().clone())
            topk = torch.topk(weight_mag.flatten(), k=int(weight_mag.nelement()*(1-args_raw.receiver_keep_perc)), largest=False)
            temp_mask = torch.ones(weight_mag.nelement())
            temp_mask[topk.indices] = 0
            mask_dict_rec[name] = temp_mask.bool().view(weight_mag.shape)
    else:
        mask_dict_rec = {}
        for name, param in team.rbot.named_parameters():
                mask_dict_rec[name] = torch_bernoulli(args_raw.receiver_keep_perc, param.shape)

for i in range(args['trainIters']):
    candidates, targets = data.getBatchData(train_np, args['batchSize'], args['distractNum'], easy_limit=args_raw.easy_limit)
    sloss, rloss, message, rewards, entropy, _, _ = team.forward(targets, candidates, False, True, True, stochastic=True)
    if i / args['resetIter'] >= args_raw.freeze_after_gen_num and args_raw.freeze_receiver:
        team.backward(sloss, rloss, rOpt=False)
    else:
        team.backward(sloss, rloss)

    sloss_l[i] = sloss
    rloss_l[i] = rloss
    trainAccuracy_l[i] = rewards.sum().item() / args['batchSize'] * 100  # reward +1 0
    entropy_l[i] = entropy

    # print intermediate results during training
    if i % 100 == 0:
        record = 'Iteration ' + str(i) \
                 + ' Sender loss ' + str(np.round(sloss_l[i], decimals=4)) \
                 + ' Recever loss ' + str(np.round(rloss_l[i], decimals=4)) \
                 + ' Training accuracy ' + str(np.round(trainAccuracy_l[i], decimals=2)) + '%\n'
        print(record)

        if not args_raw.no_wandb:
            wandb.log({'Iter': i, 'Sender Loss': np.round(sloss_l[i], decimals=4), 'Recever loss':np.round(rloss_l[i], decimals=4), 'Training accuracy':np.round(trainAccuracy_l[i], decimals=2)})
            
    if args_raw.track_examples and i%10 == 0:
        with torch.no_grad():
            _, _, msg, hard_color_rewards, _, _, _ = team.forward(hard_color_targets, hard_color_inputs, False, True, True, stochastic=False)
            _, _, _, hard_shape_rewards, _, _, _ = team.forward(hard_shape_targets, hard_shape_inputs, False, True, True, stochastic=False)
            _, _, _, easy_rewards, _, _, _ = team.forward(easy_targets, easy_inputs, False, True, True, stochastic=False)
            _, _, _, random_rewards, _, _, _ = team.forward(random_targets, random_inputs, False, True, True, stochastic=False)
            dTopo_tmp, _, _ = util.get_sender_language(team, neural=True) # calculate topo similarity before 
            
            hard_color_tracker.append(hard_color_rewards.cpu().numpy())
            hard_shape_tracker.append(hard_shape_rewards.cpu().numpy())
            easy_tracker.append(easy_rewards.cpu().numpy())
            random_tracker.append(random_rewards.cpu().numpy())
            
            message_tracker.append(msg.cpu().numpy())
            
            topo_tracker.append(dTopo_tmp)
            time_tracker.append(i)
    
    if i != 0 and i % args['resetIter'] == 0:
#         args_raw.easy_limit -= 1
        # evaluate before reset
        print('Periodically Evaluation: : ')
        print('For the ' + str(i // args['resetIter']) + 'th evaluation')  # start from 1
        with torch.no_grad():
            ind = i // args['resetIter']
            dTopo[ind], dEntropy[ind], curLangD = util.get_sender_language(team, neural=True) # calculate topo similarity before each reset
            if not args_raw.no_wandb:
                wandb.log({'Iter': i, 'Topographic Similarity': dTopo[ind]})
        if args_raw.save_weights:
            sbot_weights_all_time.append(save_weight_one_iter(team.sbot))
            rbot_weights_all_time.append(save_weight_one_iter(team.rbot))
            time_ind.append(i)
            sloss_tracker.append(sloss_l[i])
            rloss_tracker.append(rloss_l[i])
            acc_tracker.append(trainAccuracy_l[i])   
            topo_tracker.append(dTopo[ind])               

#         if ind % 10 == 0:
#             if not args_raw.reset_receiver:
#                 oldReceiver, oldrOptimizer = team.copyReceiver()

#             team.freezeSender()
#             eval_teach_speed(ind // 10 - 1, data, team)
#             team.defreezeSender()
#             if not args_raw.reset_receiver:
#                 team.rbot = oldReceiver
#                 team.rOptimizer = oldrOptimizer

        if args_raw.reset_receiver:
            team.resetReceiver()
          
            
        if args_raw.forget_sender:
            if args_raw.sender_keep_perc == 0:
                team.resetSender()
            else:
                if not args_raw.same_mask:
                    if args_raw.weight_mask:
                        mask_dict = {}
                        for name, param in team.sbot.named_parameters():
                            weight_mag = torch.abs(param.detach().clone())
                            topk = torch.topk(weight_mag.flatten(), k=int(weight_mag.nelement()*(1-args_raw.sender_keep_perc)), largest=False)
                            temp_mask = torch.ones(weight_mag.nelement())
                            temp_mask[topk.indices] = 0
                            mask_dict[name] = temp_mask.bool().view(weight_mag.shape)
                    else:
                        mask_dict = {}
                        for name, param in team.sbot.named_parameters():
                                mask_dict[name] = torch_bernoulli(args_raw.sender_keep_perc, param.shape)
                if args_raw.reset_to_zero:
                    for name, param in team.sbot.named_parameters():
                        param.data *= mask_dict[name].float().to('cuda')
                else:
                    metric_dict = {}
                    for name, param in team.sbot.named_parameters():
                        metric_dict[name] = param.detach().clone()

                    for name, param in team.sbot.named_children():
                        param.reset_parameters()

                    for name, param in team.sbot.named_parameters():
                        param.data[mask_dict[name]] = metric_dict[name][mask_dict[name]]
                    
        if args_raw.forget_receiver:
            if args_raw.receiver_keep_perc == 0:
                team.resetReceiver()
            else:
                if not args_raw.same_mask:
                    if args_raw.weight_mask:
                        mask_dict_rec = {}
                        for name, param in team.rbot.named_parameters():
                            weight_mag = torch.abs(param.detach().clone())
                            topk = torch.topk(weight_mag.flatten(), k=int(weight_mag.nelement()*(1-args_raw.receiver_keep_perc)), largest=False)
                            temp_mask = torch.ones(weight_mag.nelement())
                            temp_mask[topk.indices] = 0
                            mask_dict_rec[name] = temp_mask.bool().view(weight_mag.shape)
                    else:
                        mask_dict_rec = {}
                        for name, param in team.rbot.named_parameters():
                                mask_dict_rec[name] = torch_bernoulli(args_raw.receiver_keep_perc, param.shape)
                if args_raw.reset_to_zero:
                    for name, param in team.rbot.named_parameters():
                        param.data *= mask_dict_rec[name].float().to('cuda')
                else:
                    metric_dict = {}
                    for name, param in team.rbot.named_parameters():
                        metric_dict[name] = param.detach().clone()

                    for name, param in team.rbot.named_children():
                        param.reset_parameters()

                    for name, param in team.rbot.named_parameters():
                        param.data[mask_dict_rec[name]] = metric_dict[name][mask_dict_rec[name]]
        
        if args_raw.freeze_receiver and i / args['resetIter'] >= args_raw.freeze_after_gen_num:
            team.freezeReceiver()
        print('similarity after forget')
        with torch.no_grad():
            dTopo_new, _, _ = util.get_sender_language(team, neural=True)
            if not args_raw.no_wandb:
                wandb.log({'Iter': i, 'New Topographic Similarity': dTopo_new})    
            
        if args_raw.track_examples:
            with torch.no_grad():
                _, _, msg, hard_color_rewards, _, _, _ = team.forward(hard_color_targets, hard_color_inputs, False, True, True, stochastic=False)
                _, _, _, hard_shape_rewards, _, _, _ = team.forward(hard_shape_targets, hard_shape_inputs, False, True, True, stochastic=False)
                _, _, _, easy_rewards, _, _, _ = team.forward(easy_targets, easy_inputs, False, True, True, stochastic=False)
                _, _, _, random_rewards, _, _, _ = team.forward(random_targets, random_inputs, False, True, True, stochastic=False)
                
                hard_color_tracker.append(hard_color_rewards.cpu().numpy())
                hard_shape_tracker.append(hard_shape_rewards.cpu().numpy())
                easy_tracker.append(easy_rewards.cpu().numpy())
                random_tracker.append(random_rewards.cpu().numpy())
                
                message_tracker.append(msg.cpu().numpy())

                topo_tracker.append(dTopo_new)
                time_tracker.append(i)
    

print('After training for ' + str(args['trainIters']) + ' iterations')
with torch.no_grad():
    dTopo[-1], dEntropy[-1], langD = util.get_sender_language(team, neural=True) # evaluate all group performance
    np.save(args['fname'] + '/langDict', langD)
    
if not args_raw.no_wandb:
    wandb.log({'Iter': i+1, 'Topographic Similarity': dTopo[-1]})

# speed of teaching the language to a new listener after determinized
# make params untrainable, testing if sender is not learning

# team.freezeSender()
# eval_teach_speed(args['resetNum'] // 10 - 1, data, team)

# np.save(args['fname'] + '/sloss', sloss_l)
# np.save(args['fname'] + '/rloss', rloss_l)
# np.save(args['fname'] + '/trainAcc', trainAccuracy_l)
# np.save(args['fname'] + '/entropy', entropy_l)
# np.save(args['fname'] + '/dTopo', dTopo)
# np.save(args['fname'] + '/dEntropy', dEntropy)
# np.save(args['fname'] + '/evalAcc', evalAcc_l)


if args_raw.track_examples:
    hard_color_tracker = np.vstack(hard_color_tracker)
    hard_shape_tracker = np.vstack(hard_shape_tracker)
    easy_tracker = np.vstack(easy_tracker)
    random_tracker = np.vstack(random_tracker)
    
    message_tracker = np.stack(message_tracker)   

    with h5py.File(os.path.join(args['fname'], 'seed%d_example_tracker.h5'%args['seed']), 'w') as f:
        f.attrs['resetIter'] = args_raw.resetIter
        f.attrs['trainIters'] = args_raw.trainIters
        f.attrs['resetNum'] = args_raw.resetNum

        a_dset = f.create_dataset('hard_color_tracker', hard_color_tracker.shape)
        a_dset[:] = hard_color_tracker
        b_dset = f.create_dataset('hard_shape_tracker', hard_shape_tracker.shape)
        b_dset[:] = hard_shape_tracker
        c_dset = f.create_dataset('easy_tracker', easy_tracker.shape)
        c_dset[:] = easy_tracker
        d_dset = f.create_dataset('random_tracker', random_tracker.shape)
        d_dset[:] = random_tracker
        
        topo_tracker = np.array(topo_tracker)
        j_dset = f.create_dataset('topo_tracker', topo_tracker.shape)
        j_dset[:] = topo_tracker        

        e_dset = f.create_dataset('all_color_hard_examples', all_color_hard_examples.shape)
        e_dset[:] = all_color_hard_examples
        f_dset = f.create_dataset('all_shape_hard_examples', all_shape_hard_examples.shape)
        f_dset[:] = all_shape_hard_examples
        g_dset = f.create_dataset('all_easy_examples', all_easy_examples.shape)
        g_dset[:] = all_easy_examples
        h_dset = f.create_dataset('all_random_examples', all_random_examples.shape)
        h_dset[:] = all_random_examples
        
        k_dset = f.create_dataset('message_tracker', message_tracker.shape)
        k_dset[:] = message_tracker
        
        time_tracker = np.array(time_tracker)
        i_dset = f.create_dataset('time_tracker', time_tracker.shape)
        i_dset[:] = time_tracker
        
if args_raw.save_weights:
    sbot_weights_all_time = np.vstack(sbot_weights_all_time)
    rbot_weights_all_time = np.vstack(rbot_weights_all_time)
    #save metrics
    with h5py.File(os.path.join(args['fname'], 'seed%d_weight_trajectory.h5'%args['seed']), 'w') as f:
        s_dset = f.create_dataset('sbot_weights_all_time', sbot_weights_all_time.shape)
        s_dset[:] = sbot_weights_all_time
        r_dset = f.create_dataset('rbot_weights_all_time', rbot_weights_all_time.shape)
        r_dset[:] = rbot_weights_all_time

        t_dset = f.create_dataset('time_ind', np.array(time_ind).shape)
        t_dset[:] = np.array(time_ind)
        sl_dset = f.create_dataset('sloss_tracker', np.array(sloss_tracker).shape)
        sl_dset[:] = np.array(sloss_tracker)
        rl_dset = f.create_dataset('rloss_tracker', np.array(rloss_tracker).shape)
        rl_dset[:] = np.array(rloss_tracker)
        acc_dset = f.create_dataset('acc_tracker', np.array(acc_tracker).shape)
        acc_dset[:] = np.array(acc_tracker)    
        topo_dset = f.create_dataset('topo_tracker', np.array(topo_tracker).shape)
        topo_dset[:] = np.array(topo_tracker)            
