import argparse,os
from utils import loading_contact_map,loading_seq,get_label,loading_item_data,PFPDataset,collate
from torch.utils.data import DataLoader
from model import Model_Net
from featureGen import ProSEMT
import torch,json
import torch.nn as nn
import torch.optim as optim
from utils import get_results
import warnings

def train(args):
    warnings.filterwarnings("ignore")
    device = torch.device(args.device)
    model_feature = ProSEMT.load_pretrained().to(device).eval()

    for data_source in args.data:
        if data_source == 'pdb':
            print('loading contact maps for pdb structures...')
            dict_contact_pdb = loading_contact_map(data_source,args.threshold)
            print('loading seq for pdb structures...')
            dict_seq_pdb = loading_seq(data_source)
        elif data_source == 'alpha':
            print('loading contact maps for alphafold structures...')
            dict_contact_alpha = loading_contact_map(data_source,args.threshold)
            print('loading seq for alphafold structures...')
            dict_seq_alpha = loading_seq(data_source)

    for func in args.func:
        label,label_num = get_label(func)
        train_item = loading_item_data(func,'train')
        valid_item = loading_item_data(func,'valid')
        test_item = loading_item_data(func,'test')
        for data_source in args.data:
            print('############################################################')
            print('training for '+str(func)+' using '+args.net+','+str(data_source)+' data...')
            m_AUPR = -1.0
            if data_source == 'pdb':
                train_data = PFPDataset(train_data_X=train_item, train_contactmap_X=dict_contact_pdb,train_feature_matrix_X=dict_seq_pdb, train_data_Y=label,model_feature=model_feature,device=device)

                valid_data = PFPDataset(train_data_X=valid_item, train_contactmap_X=dict_contact_pdb,train_feature_matrix_X=dict_seq_pdb, train_data_Y=label,model_feature=model_feature,device=device)

                #test_data = PFPDataset(train_data_X=test_item, train_contactmap_X=dict_contact_pdb,train_feature_matrix_X=dict_seq_pdb, train_data_Y=label,model_feature=model_feature,device=device)

            elif data_source == 'alpha':
                train_data = PFPDataset(train_data_X=train_item, train_contactmap_X=dict_contact_alpha,train_feature_matrix_X=dict_seq_alpha, train_data_Y=label,model_feature=model_feature,device=device)

                valid_data = PFPDataset(train_data_X=valid_item, train_contactmap_X=dict_contact_alpha,train_feature_matrix_X=dict_seq_alpha, train_data_Y=label,model_feature=model_feature,device=device)

                #test_data = PFPDataset(train_data_X=test_item, train_contactmap_X=dict_contact_alpha,train_feature_matrix_X=dict_seq_alpha, train_data_Y=label,model_feature=model_feature,device=device)

            dataset_train = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate, drop_last=False)
            dataset_valid = DataLoader(valid_data, batch_size=64, shuffle=False, collate_fn=collate, drop_last=False)
            #dataset_test = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate, drop_last=False)
            model = Model_Net(output_dim=label_num,net=args.net,hidden_dim=args.hidden_dim,pool=args.pool,dropout=args.dropout).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            bceloss = nn.BCELoss()
            for e in range(args.Epoch):
                model = model.train()
                for batch_idx, batch in enumerate(dataset_train):
                    data_prot = batch.to(device)
                    optimizer.zero_grad()
                    output = model(data_prot)
                    loss = bceloss(output, data_prot.label)
                    loss.backward()
                    optimizer.step()
                    #a = model(batch)

                model = model.eval()
                total_preds = torch.Tensor()
                total_labels = torch.Tensor()
                with torch.no_grad():
                    for batch_idx,batch in enumerate(dataset_valid):
                        data_prot = batch.to(device)
                        output_valid = model(data_prot)
                        total_preds = torch.cat((total_preds, output_valid.cpu()), 0)
                        total_labels = torch.cat((total_labels, data_prot.label.cpu()), 0)
                    loss_valid = bceloss(total_preds, total_labels)

                perf = get_results(total_labels.cpu().numpy(), total_preds.cpu().numpy())
                if perf['all']['m-aupr']>m_AUPR:
                    m_AUPR = perf['all']['m-aupr']
                    # torch.save(model.state_dict(),'./data_collect/'+str(func)+'/model/'+str(data_source)+'_'+str(func)+'_'+args.net+'_'+str(args.hidden_dim)+'_'+str(m_AUPR)+'_'+args.pool+'_'+str(args.dropout)+'_'+str(args.threshold)+'.pkl')

                    torch.save(model.state_dict(),'./data_collect/' + str(func) + '/model/' + str(data_source) + '_' + str(func) + '_' + args.net + '_' + str(args.hidden_dim) + '_' + args.pool + '_' + str(args.dropout) + '_' + str(args.threshold) + '.pkl')
                    if args.save_results:
                        with open('./data_collect/' + str(func) + '/' + str(data_source) + '_' + str(func) + '.json','w') as f:
                            json.dump(perf, f)
                print('Epoch ' + str(e + 1) + '\tTrain loss:\t', loss.cpu().detach().numpy(),'\tValid loss:\t', loss_valid.numpy(),'\tM-AUPR:\t',perf['all']['M-aupr'],'\tm-AUPR:\t',perf['all']['m-aupr'],'\tF-max:\t',perf['all']['F-max'])
                #print(total_preds.shape,total_labels.shape)

            #perf = get_results(total_labels.cpu().numpy(), total_preds.cpu().numpy())






if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--func', type=lambda s: [item for item in s.split(",")],
                        default=['mf','bp','cc'], help="list of func to predict.")#,'bp','cc'
    parser.add_argument('--data', type=lambda s: [item for item in s.split(",")],
                        default=['pdb', 'alpha'], help="data source.")#, 'alpha'
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help="learning rate.")
    parser.add_argument('--Epoch', type=int,
                        default=50, help="epoch for training.")
    parser.add_argument('--save_results', type=int, default=0, help="whether to save the performance results")
    parser.add_argument('--save_model', type=int, default=1, help="whether to save the model parameters")
    parser.add_argument('--net', type=str, default='GCN', help="GCN or GAT for model")
    parser.add_argument('--hidden_dim', type=int, default=512, help="hidden dim for linear")
    parser.add_argument('--device', type=str, default='cuda:1', help="cuda for model")
    parser.add_argument('--pool', type=str, default='gap-gmp', help="pool for model(gep、gap、gmp、gap-gep、gap-gmp)")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout for model")
    parser.add_argument('--threshold', type=float, default=8.0, help="distance threshold between residues")
    args = parser.parse_args()
    print(args)
    train(args)