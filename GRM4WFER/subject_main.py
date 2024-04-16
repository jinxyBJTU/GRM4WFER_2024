from args import get_args
import torch
from printScore import valence3_PrintScore as ps
from models.my_config import Config
from dataloader import data_generator, Subject_kFoldGenerator
from models.GRM4WFER import Mine_case, Mine_ceap
import numpy as np
import time
import os
from utils import Inter_Sim_withinK, Intra_Sim_withinK

def sampling_from_repara(mu, sigma):
    epsilon = torch.randn_like(mu).to(mu.device)
    return mu+epsilon*torch.sqrt(sigma)

def model_train(args, configs, model, optimizer, criterion_cls, train_loader, device):
    total_loss_list = []
    total_feature_list = []
    total_uncer_list = []
    total_fine_list = []
    weak_acc = []
    model.train()   

    for batch_idx, (all_signals, all_fine_labels, all_weak_labels, all_lengths) in enumerate(train_loader):
        all_signals = all_signals.to(device)            # 1, 8, 30, 3, 100
        all_fine_labels = all_fine_labels.to(device)    # 1, 8, 30
        all_weak_labels = all_weak_labels.to(device)    # 1, 8, 30
        all_lengths = all_lengths.to(device)            # 1, 8

        provoke_criterion = torch.nn.HingeEmbeddingLoss()

        signals = all_signals[0]
        weak_labels = all_weak_labels[0,:,0]
        fine_labels = all_fine_labels[0]       
        cur_lengths = all_lengths[0]

        after_gcn_features, trail_features, trail_uncers, logit_of_bag = model( signals, weak_labels, cur_lengths )  # (8,30,128),(8,30,128) 
        
        pred_of_bag = torch.argmax(logit_of_bag, dim=-1)
        celoss = criterion_cls(logit_of_bag, weak_labels)
        
        # ins_celoss = criterion_cls(logit_of_instances, fine_labels.reshape(-1))
        
        if configs.use_dis_constraint and (1 in weak_labels and (0 in weak_labels or 2 in weak_labels )):
            if configs.use_uncertaiy == False:
                assert configs.dis_type == 'Elu'

            num_of_trail, num_of_instance, num_of_feas = trail_features.shape
            num_of_neutral = int(torch.sum(weak_labels==1))
            num_of_simuli = num_of_trail - num_of_neutral
            
            normals_index = all_weak_labels[0,:]==1
            normals_index = normals_index.unsqueeze(-1).repeat(1,1,num_of_feas)
            anamoly_index = ~normals_index

            normal_features = torch.masked_select(trail_features, normals_index).reshape(num_of_neutral,num_of_instance,-1)
            anamoly_features = torch.masked_select(trail_features, anamoly_index).reshape(num_of_simuli,num_of_instance,-1)
            normal_uncers = torch.masked_select(trail_uncers, normals_index).reshape(num_of_neutral,num_of_instance,-1)
            anamoly_uncers = torch.masked_select(trail_uncers, anamoly_index).reshape(num_of_simuli,num_of_instance,-1)
            
            # mean_simi_within_normal = Intra_Sim_withinK(normal_features)
            # mean_simi_within_cross = Inter_Sim_withinK(anamoly_features, normal_features)
            if 'CASE' in args.data_path:
                mean_simi_within_normal = Intra_Sim_withinK(configs.dis_type, normal_features, normal_uncers, cur_lengths[weak_labels==1])
                mean_simi_within_cross = Inter_Sim_withinK(configs.dis_type, anamoly_features, normal_features, anamoly_uncers, normal_uncers, cur_lengths[weak_labels!=1], cur_lengths[weak_labels==1])
            else:
                mean_simi_within_normal = Intra_Sim_withinK(configs.dis_type, normal_features, normal_uncers)
                mean_simi_within_cross = Inter_Sim_withinK(configs.dis_type, anamoly_features, normal_features, anamoly_uncers, normal_uncers)
            
            y_true = torch.tensor([-1], dtype=torch.float32).to(signals.device)
            y_pred_scores = mean_simi_within_normal - mean_simi_within_cross
            provoke_loss = provoke_criterion(y_pred_scores, y_true)
            
            total_loss = celoss + provoke_loss
        else:
            total_loss = celoss 
            
        loss_items = torch.tensor([total_loss.item(), celoss.item()]).unsqueeze(0)
        total_loss_list.append(loss_items)
        total_feature_list.append(after_gcn_features.detach().cpu().numpy())
        total_uncer_list.append(trail_uncers.detach().cpu().numpy())
        total_fine_list.append(fine_labels.detach().cpu().numpy())

        weak_acc.append( weak_labels.eq( pred_of_bag.detach() ).float().mean() )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
   
    total_loss_list = torch.cat(total_loss_list, dim=0)
    mean_loss_list = torch.mean(total_loss_list, dim=0).tolist()
    total_mean_loss = mean_loss_list[0]
    weak_acc = torch.tensor(weak_acc).mean().item() 

    total_feature_list = np.stack(total_feature_list, axis=0)
    total_uncer_list = np.stack(total_uncer_list, axis=0)
    total_fine_list = np.stack(total_fine_list, axis=0)
    
    return total_mean_loss, mean_loss_list, weak_acc, total_feature_list, total_uncer_list, total_fine_list

def model_evaluate(args, configs, model, test_loader, device):
    model.eval()
    total_loss = []
    fine_acc = []
    weak_acc = []

    criterion = torch.nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    out_list = []
    fine_trgs_list = []
    weak_trgs_list = []
    feature_list = []

    with torch.no_grad():
        for batch_idx, (all_signals, all_fine_labels, all_weak_labels, all_lengths) in enumerate(test_loader):
            all_signals = all_signals.to(device)            # 1, 8, 30, 3, 100          
            all_fine_labels = all_fine_labels.to(device)    # 1, 8, 30
            all_weak_labels = all_weak_labels.to(device)    # 1, 8, 30
            all_lengths = all_lengths.to(device)            # 1, 8

            signals = all_signals[0]
            weak_labels = all_weak_labels[0,:]  # 8,30
            fine_labels = all_fine_labels[0]    # 8,30
            cur_lengths = all_lengths[0]

            features, instance_gains = model(signals, all_weak_labels[0,:,0], cur_lengths, is_testing = True)
            # 8,30 / 8,98 
            if 'CASE' in args.data_path:
                for trail_idx in range(len(instance_gains)):
                    cur_weak = weak_labels[trail_idx][:cur_lengths[trail_idx]]
                    cur_fine = fine_labels[trail_idx][:cur_lengths[trail_idx]]

                    cur_instance_gains = instance_gains[trail_idx][:cur_lengths[trail_idx]]
                    pred_of_instances = torch.zeros_like(cur_instance_gains, dtype = torch.long)

                    mean_gains = torch.mean( cur_instance_gains )                
                    pred_of_instances[cur_instance_gains<mean_gains] = 1
                    pred_of_instances[cur_instance_gains>=mean_gains] = cur_weak[cur_instance_gains>=mean_gains]
                   
                    fine_acc.append(cur_fine.eq(pred_of_instances.detach()).float().mean())
                    outs = np.append(outs, pred_of_instances.cpu().numpy())
                    trgs = np.append(trgs, cur_fine.data.cpu().numpy())
                    
                    out_list.append(pred_of_instances.cpu().numpy())
                    fine_trgs_list.append(cur_fine.data.cpu().numpy())
                    weak_trgs_list.append(cur_weak.data.cpu().numpy())

            else:
                pred_of_instances = torch.zeros_like(instance_gains, dtype = torch.long)
                
                mean_gains = torch.mean( instance_gains, dim=-1, keepdim=True )                
                pred_of_instances[instance_gains<mean_gains] = 1
                pred_of_instances[instance_gains>=mean_gains] = weak_labels[instance_gains>=mean_gains] 
                
                fine_acc.append(fine_labels.eq(pred_of_instances.detach()).float().mean())
                
                outs = np.append(outs, pred_of_instances.cpu().numpy())
                trgs = np.append(trgs, fine_labels.data.cpu().numpy())
                
                out_list.append(pred_of_instances.cpu().numpy())
                fine_trgs_list.append(fine_labels.data.cpu().numpy())
                weak_trgs_list.append(weak_labels.data.cpu().numpy())

    fine_acc = torch.tensor(fine_acc).mean().item() 
    weak_acc = torch.tensor(weak_acc).mean() 

    return fine_acc, outs, trgs, [features.cpu().numpy(), instance_gains.cpu().numpy(), weak_labels.cpu().numpy(), fine_labels.cpu().numpy(), cur_lengths.cpu().numpy()]
   
def main(args):

    print('*'*25 + ' seed = ' + str(args.seed) + '*'*25 )
    device = torch.device( 'cuda:{}'.format(args.cuda) )
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if 'CASE' in args.data_path:
        max_length_trails = 98
        input_dim = 3
        total_subjects = 30
        num_of_trail = 8
        signal_length = 100
        
    elif '360' in args.data_path:
        max_length_trails = 30
        input_dim = 3
        total_subjects = 32
        num_of_trail = 8
        signal_length = 100

    configs = Config()
    configs.batch_size = max_length_trails

    Model_name = f'UQ{int(configs.use_uncertaiy)}_DC{int(configs.use_dis_constraint)}_GCN{int(configs.use_crossTrail_gcn)}_|_rep{configs.repeat_times}_ratio{configs.drop_ratio}_dt-{configs.dis_type}_graph-{configs.graph_type}'
    
    base_path = f'result_{Model_name}/' + args.data_path.split('/')[-2] + f'/{args.task}_Subject_{args.seed}' 
    final_csv = f'result_{Model_name}/' + args.data_path.split('/')[-2] + f'/{args.task}_Subject_{args.seed}.csv' 
    result_name = f'result_{Model_name}/' + args.data_path.split('/')[-2] + f'/{args.task}_results_{args.seed}.npz' 

    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    total_acc = []
    total_f1 = []
    total_kappa = []
    total_output_lists = []

    total_signals, total_fine_labels, total_weak_labels, total_lengths = data_generator(args.data_path, args.task, total_subjects, max_length_trails)
    
    Data_Loader = Subject_kFoldGenerator(total_signals, total_fine_labels, total_weak_labels, total_lengths)

    for subject_idx in range(total_subjects):
        print()
        print('*'*25 + f'This is Subject-{subject_idx+1}' + '*'*25 )

        log_save_file = base_path + f'/Subject_{subject_idx+1}_log' 

        with open(log_save_file , 'w') as f:
            f.writelines( f'This is Subject-{subject_idx+1}\n' )
            f.writelines( '\n' )

        if 'CASE' in args.data_path:
            model = Mine_case(configs = configs, input_dim=input_dim, signal_length=signal_length, batch_size = configs.batch_size, num_of_classes=args.n_classes).to(device)
        else:
            model = Mine_ceap(configs = configs, input_dim=input_dim, signal_length=signal_length, batch_size = configs.batch_size, num_of_classes=args.n_classes).to(device)

        train_loader, test_loader = Data_Loader.getFold( subject_idx, log_save_file )

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr) 
        criterion_cls = torch.nn.CrossEntropyLoss().to(device)

        total_param = 0
        for param_tensor in model.state_dict():
            total_param += np.prod(model.state_dict()[param_tensor].size())
        print('final_model\'s total params:' + str(total_param))

        with open(log_save_file , 'a') as f:
            f.writelines('final_model total params:' + str(total_param) + '\n')

        best_train_loss = np.inf
        not_improved_count = 0

        log_sentence = f'\nSubject:{subject_idx+1}\t Train Loss \t | Train Acc \t | Test Acc \t | Train Time'
        print(log_sentence)

        with open(log_save_file , 'a') as f:
            f.writelines( log_sentence )
            f.writelines( '\n' )

        # Epoch : {epoch}\n
        best_acc = 0
        for epoch in range(1, configs.train_epochs + 1):
            # Train and validate
            start_time = time.time()

            mean_train_loss, train_loss_list, train_fine_acc, train_feature_list, train_uncer_list, train_fine_list = model_train(args, configs, model, optimizer, criterion_cls, train_loader, device)
            
            after_train = time.time()

            test_fine_acc, outs, trgs, output_lists  = model_evaluate(args, configs, model, test_loader, device)
            
            log_sentence = f'Epoch {epoch}: \t'
            for loss_item in train_loss_list:
                log_sentence += f'{loss_item:.4f}\t'
            log_sentence += f'\t | {train_fine_acc:2.4f} \t | {test_fine_acc:2.4f} \t | {after_train-start_time:<7.2f}s'

            print(log_sentence)
            
            with open(log_save_file , 'a') as f:
                f.writelines( log_sentence )
                f.writelines( '\n' )
            
            if test_fine_acc >= best_acc:
                best_acc = test_fine_acc
                best_trgs = trgs
                best_outs = outs
                best_output_lists = output_lists

            if mean_train_loss < best_train_loss:
                best_train_loss = mean_train_loss
                not_improved_count = 0
            else:
                not_improved_count += 1

            if not_improved_count == configs.early_stop_steps:
                print("early stop")
                break

        if configs.valid_best:
            acc, f1, kappa = ps(best_trgs, best_outs, savePath=log_save_file, average='weighted')
        else:
            acc, f1, kappa = ps(trgs, outs, savePath=log_save_file, average='weighted')

        total_acc.append(acc)
        total_f1.append(f1)
        total_kappa.append(kappa)
        total_output_lists.append(best_output_lists)

    total_features = []
    total_gains = []
    total_weaks = []
    total_fines = []
    total_lengths = []
    num_of_subjects = len(total_output_lists)
    for subject_idx in range(num_of_subjects):    
        total_features.append(total_output_lists[subject_idx][0])
        total_gains.append(total_output_lists[subject_idx][1])
        total_weaks.append(total_output_lists[subject_idx][2])
        total_fines.append(total_output_lists[subject_idx][3])
        total_lengths.append(total_output_lists[subject_idx][4])

    total_features = np.stack(total_features,axis=0)
    total_gains = np.stack(total_gains,axis=0)
    total_weaks = np.stack(total_weaks,axis=0)
    total_fines = np.stack(total_fines,axis=0)
    total_lengths = np.stack(total_lengths,axis=0)
    print(total_features.shape)
    print(total_gains.shape)
    print(total_weaks.shape)
    print(total_fines.shape)
    print(total_lengths.shape)
    print()
    np.savez_compressed(
            result_name ,
            test_features = total_features,
            instance_gains = total_gains,
            weak_labels = total_weaks,
            fine_labels = total_fines,
            total_lengths = total_lengths,
            )

    mean_acc = np.mean(np.array(total_acc))  
    mean_f1 = np.mean(np.array(total_f1))  
    mean_kappa = np.mean(np.array(total_kappa))  

    print('Total Performance:')
    print(f'Acc:{mean_acc:.4f}\t F1-S:{mean_f1:.4f}\t Kappa:{mean_kappa:.4f}')

    with open(final_csv, 'w') as f:
        sentence = f'Acc:{mean_acc:.4f}\t F1-S:{mean_f1:.4f}\t Kappa:{mean_kappa:.4f}\n'
        f.writelines(sentence)
        f.writelines('\n')

        idx = 0
        for subject_idx in range(total_subjects):
            sentence = f'Subject:{subject_idx}: {total_acc[idx]:.4f}\t {total_f1[idx]:.4f}\t {total_kappa[idx]:.4f}\n'
            f.writelines(sentence)
            idx += 1

if __name__ == '__main__':
    main(get_args())