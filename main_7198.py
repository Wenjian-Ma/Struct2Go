import argparse,os
from utils import PFPDataset,collate
from torch.utils.data import DataLoader
from model import Model_Net
from featureGen import ProSEMT
import torch,json
import torch.nn as nn
import torch.optim as optim
from utils import get_results
import warnings
from tqdm import tqdm
import numpy as np


def loading_item_data(func,name,data_source):
    item = []
    with open('./data_collect/'+func+'/'+name+'_data_new.txt','r') as f:
        for line in f:
            line = line.strip('\n')
            item.append(line)


    return item


def get_label(func,data_source):
    if func == 'mf':
        mf_dict = {}
        mf_func = []
        mf_label_dict = {}
        with open('./data_collect/amplify_samples/Go_label_6582.txt', 'r') as f:
            for line in f:
                if '>' in line:
                    pdb_chain_uid = line[1:].strip('\n')
                elif 'mf' in line:
                    if 'mf:' == line.strip('\n'):
                        continue
                    else:
                        mf_func_list = line[3:].strip().split('\t')
                        mf_dict[pdb_chain_uid] = mf_func_list

        with open('/media/administrator/SEAGATE-4T/public/PFP/Struct2Go/data_collect/Go_label.txt', 'r') as f:
            for line in f:
                if '>' in line:
                    pdb_chain_uid = line[1:].strip('\n')
                elif 'mf' in line:
                    if 'mf:' == line.strip('\n'):
                        continue
                    else:
                        mf_func_list = line[3:].strip().split('\t')
                        mf_dict[pdb_chain_uid] = mf_func_list
        with open('./data_collect/mf/mf_label.txt','r') as f:
            for line in f:
                line = line.strip('\n')
                mf_func.append(line)
        for i in mf_dict.keys():
            label = np.zeros(len(mf_func))
            for j in mf_dict[i]:
                if j in mf_func:
                    index = mf_func.index(j)
                    label[index] = 1
            mf_label_dict[i] = label
        return mf_label_dict,len(mf_func)
    elif func == 'bp':
        bp_dict = {}
        bp_func = []
        bp_label_dict = {}
        with open('./data_collect/amplify_samples/Go_label_6582.txt', 'r') as f:
            for line in f:
                if '>' in line:
                    pdb_chain_uid = line[1:].strip('\n')
                elif 'bp' in line:
                    if 'bp:' == line.strip('\n'):
                        continue
                    else:
                        bp_func_list = line[3:].strip().split('\t')
                        bp_dict[pdb_chain_uid] = bp_func_list

        with open('/media/administrator/SEAGATE-4T/public/PFP/Struct2Go/data_collect/Go_label.txt', 'r') as f:
            for line in f:
                if '>' in line:
                    pdb_chain_uid = line[1:].strip('\n')
                elif 'bp' in line:
                    if 'bp:' == line.strip('\n'):
                        continue
                    else:
                        bp_func_list = line[3:].strip().split('\t')
                        bp_dict[pdb_chain_uid] = bp_func_list
        with open('./data_collect/bp/bp_label.txt','r') as f:
            for line in f:
                line = line.strip('\n')
                bp_func.append(line)
        for i in bp_dict.keys():
            label = np.zeros(len(bp_func))
            for j in bp_dict[i]:
                if j in bp_func:
                    index = bp_func.index(j)
                    label[index] = 1
            bp_label_dict[i] = label
        return bp_label_dict,len(bp_func)
    elif func == 'cc':
        cc_dict = {}
        cc_func = []
        cc_label_dict = {}
        with open('./data_collect/amplify_samples/Go_label_6582.txt', 'r') as f:
            for line in f:
                if '>' in line:
                    pdb_chain_uid = line[1:].strip('\n')
                elif 'cc' in line:
                    if 'cc:' == line.strip('\n'):
                        continue
                    else:
                        cc_func_list = line[3:].strip().split('\t')
                        cc_dict[pdb_chain_uid] = cc_func_list

        with open('/media/administrator/SEAGATE-4T/public/PFP/Struct2Go/data_collect/Go_label.txt', 'r') as f:
            for line in f:
                if '>' in line:
                    pdb_chain_uid = line[1:].strip('\n')
                elif 'cc' in line:
                    if 'cc:' == line.strip('\n'):
                        continue
                    else:
                        cc_func_list = line[3:].strip().split('\t')
                        cc_dict[pdb_chain_uid] = cc_func_list
        with open('./data_collect/cc/cc_label.txt', 'r') as f:
            for line in f:
                line = line.strip('\n')
                cc_func.append(line)
        for i in cc_dict.keys():
            label = np.zeros(len(cc_func))
            for j in cc_dict[i]:
                if j in cc_func:
                    index = cc_func.index(j)
                    label[index] = 1
            cc_label_dict[i] = label
        return cc_label_dict, len(cc_func)


def loading_seq(data_source):

    if data_source == 'pdb':
        dict_seq = {}
        with open('./data_collect/pdb_mapping_seq.txt','r') as f:
            for line in tqdm(f,total=7198):
                if '>' in line:
                    pid_chain_uid = line[1:].strip('\n').split('\t')[0]
                    seq_pdb = line[1:].strip('\n').split('\t')[1]
                    dict_seq[pid_chain_uid] = seq_pdb
        with open('./data_collect/amplify_samples/seq_6582.txt','r') as f:
            for line in tqdm(f,total=6582):
                if '>' in line:
                    pid_chain_uid = line[1:].strip('\n').split('\t')[0]
                    seq_pdb = line[1:].strip('\n').split('\t')[1]
                    dict_seq[pid_chain_uid] = seq_pdb


    elif data_source == 'alpha':
        dict_seq = {}
        with open('./data_collect/alpha_mapping_seq.txt') as f:
            for line in tqdm(f,total=7198):
                if '>' in line:
                    pid_chain_uid = line[1:].strip('\n').split('\t')[0]
                    seq_pdb = line[1:].strip('\n').split('\t')[1]
                    dict_seq[pid_chain_uid] = seq_pdb
        with open('./data_collect/amplify_samples/seq_6582.txt','r') as f:
            for line in tqdm(f,total=6582):
                if '>' in line:
                    pid_chain_uid = line[1:].strip('\n').split('\t')[0]
                    seq_pdb = line[1:].strip('\n').split('\t')[1]
                    dict_seq[pid_chain_uid] = seq_pdb
    return dict_seq

def loading_contact_map(data_source,threshold):
    if data_source == 'pdb':
        contact_map_list_pdb = os.listdir('./data_collect/amplify_samples/contact_map_6582_' + str(threshold))
        contact_map_list = os.listdir('./data_collect/contact_map_dense_pdb_' + str(threshold))
        dict_contact_map = {}
        for item in tqdm(contact_map_list_pdb):
            pid_chain_uid = item.strip().split('.')[0]
            dict_contact_map[pid_chain_uid] = np.load('./data_collect/amplify_samples/contact_map_6582_' + str(threshold)+'/'+item)
        for item in tqdm(contact_map_list):
            pid_chain_uid = item.strip().split('.')[0]
            dict_contact_map[pid_chain_uid] = np.load('./data_collect/contact_map_dense_pdb_'+str(threshold)+'/'+item)
    elif data_source == 'alpha':
        contact_map_list_pdb = os.listdir('./data_collect/amplify_samples/contact_map_6582_' + str(threshold))
        contact_map_list_alpha = os.listdir('./data_collect/contact_map_dense_alphafold_'+str(threshold))
        dict_contact_map = {}
        for item in tqdm(contact_map_list_pdb):
            pid_chain_uid = item.strip().split('.')[0]
            dict_contact_map[pid_chain_uid] = np.load('./data_collect/amplify_samples/contact_map_6582_' + str(threshold)+'/'+item)
        for item in tqdm(contact_map_list_alpha):
            pid_chain_uid = item.strip().split('.')[0]
            dict_contact_map[pid_chain_uid] = np.load('./data_collect/contact_map_dense_alphafold_' + str(threshold) + '/' + item)
    return dict_contact_map


def train(args):
    warnings.filterwarnings("ignore")
    device = torch.device(args.device)
    model_feature = ProSEMT.load_pretrained().to(device).eval()

    for data_source in args.data:
        if data_source == 'pdb':
            print('loading contact maps for 7198 pdb structures and 6581 pdb structures...')
            dict_contact_pdb = loading_contact_map(data_source,args.threshold)
            print('loading seq for 7198 pdb structures and 6581 pdb structures...')
            dict_seq_pdb = loading_seq(data_source)
        elif data_source == 'alpha':
            print('loading contact maps for 7198 alphafold structures and 6581 pdb structures...')
            dict_contact_alpha = loading_contact_map(data_source,args.threshold)
            print('loading seq for 7198 alphafold structures and 6581 pdb structures...')
            dict_seq_alpha = loading_seq(data_source)
    # print(len(dict_contact_pdb.keys()))
    # print(len(dict_seq_pdb.keys()))
    # print(len(dict_contact_alpha.keys()))
    # print(len(dict_seq_alpha.keys()))
    # exit()
    for func in args.func:
        for data_source in args.data:
            label, label_num = get_label(func,data_source)
            train_item = loading_item_data(func, 'train',data_source)
            print('train data:\t',len(train_item))
            valid_item = loading_item_data(func, 'valid',data_source)
            print('valid data:\t',len(valid_item))
            test_item = loading_item_data(func, 'test',data_source)
            print('test data:\t',len(test_item))
            print('############################################################')
            print('training for '+str(func)+' using '+args.net+','+str(data_source)+' data...')
            m_AUPR = -1.0
            if data_source == 'pdb':
                train_data = PFPDataset(train_data_X=train_item, train_contactmap_X=dict_contact_pdb,train_feature_matrix_X=dict_seq_pdb, train_data_Y=label,model_feature=model_feature,device=device)

                valid_data = PFPDataset(train_data_X=valid_item, train_contactmap_X=dict_contact_pdb,train_feature_matrix_X=dict_seq_pdb, train_data_Y=label,model_feature=model_feature,device=device)

                test_data = PFPDataset(train_data_X=test_item, train_contactmap_X=dict_contact_pdb,train_feature_matrix_X=dict_seq_pdb, train_data_Y=label,model_feature=model_feature,device=device)

            elif data_source == 'alpha':
                train_data = PFPDataset(train_data_X=train_item, train_contactmap_X=dict_contact_alpha,train_feature_matrix_X=dict_seq_alpha, train_data_Y=label,model_feature=model_feature,device=device)

                valid_data = PFPDataset(train_data_X=valid_item, train_contactmap_X=dict_contact_alpha,train_feature_matrix_X=dict_seq_alpha, train_data_Y=label,model_feature=model_feature,device=device)

                test_data = PFPDataset(train_data_X=test_item, train_contactmap_X=dict_contact_alpha,train_feature_matrix_X=dict_seq_alpha, train_data_Y=label,model_feature=model_feature,device=device)

            dataset_train = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate, drop_last=False)
            dataset_valid = DataLoader(valid_data, batch_size=64, shuffle=False, collate_fn=collate, drop_last=False)
            dataset_test = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate, drop_last=False)
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
                # if perf['all']['m-aupr']>m_AUPR:
                m_AUPR = perf['all']['m-aupr']
                    # torch.save(model.state_dict(),'./data_collect/'+str(func)+'/model/'+str(data_source)+'_'+str(func)+'_'+args.net+'_'+str(args.hidden_dim)+'_'+str(m_AUPR)+'_'+args.pool+'_'+str(args.dropout)+'_'+str(args.threshold)+'.pkl')

                torch.save(model.state_dict(), './data_collect/' + str(func) + '/model/' + str(data_source) + '_' + str(func) + '_' + args.net + '_' + str(args.hidden_dim) + '_' + args.pool + '_' + str(args.dropout) + '_' + str(args.threshold) +'_' +str(m_AUPR) + '.pkl')

                print('Epoch ' + str(e + 1) + '\tTrain loss:\t', loss.cpu().detach().numpy(),'\tValid loss:\t', loss_valid.numpy(),'\tM-AUPR:\t',perf['all']['M-aupr'],'\tm-AUPR:\t',perf['all']['m-aupr'],'\tF-max:\t',perf['all']['F-max'])
                #print(total_preds.shape,total_labels.shape)

            #perf = get_results(total_labels.cpu().numpy(), total_preds.cpu().numpy())






if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--func', type=lambda s: [item for item in s.split(",")],
                        default=['mf','bp','cc'], help="list of func to predict.")#'mf','bp',  ,'bp','cc'
    parser.add_argument('--data', type=lambda s: [item for item in s.split(",")],
                        default=['pdb','alpha'], help="data source.")#'pdb_no_amplifed','pdb_amplifed'
    parser.add_argument('--learning_rate', type=float,
                        default=0.0001, help="learning rate.")
    parser.add_argument('--Epoch', type=int,
                        default=50, help="epoch for training.")
    parser.add_argument('--save_results', type=int, default=0, help="whether to save the performance results")
    parser.add_argument('--save_model', type=int, default=1, help="whether to save the model parameters")
    parser.add_argument('--net', type=str, default='GCN', help="GCN or GAT for model")
    parser.add_argument('--hidden_dim', type=int, default=512, help="hidden dim for linear")
    parser.add_argument('--device', type=str, default='cuda:0', help="cuda for model")
    parser.add_argument('--pool', type=str, default='gap-gmp', help="pool for model(gep、gap、gmp、gap-gep、gap-gmp)")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout for model")
    parser.add_argument('--threshold', type=float, default=8.0, help="distance threshold between residues")
    args = parser.parse_args()
    print(args)
    train(args)