from model import Model_Net
import argparse,os,warnings
import torch,numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import loading_item_data,PFPDataset,collate,get_results
from featureGen import ProSEMT
from tqdm import tqdm

def loading_seq(data_source):
    dict_seq = {}
    if data_source == 'pdb':
        with open('./data_collect/external_pdb_mapping_seq.txt','r') as f:
            for line in tqdm(f,total=625):
                if '>' in line:
                    pid_chain_uid = line[1:].strip('\n').split('\t')[0]
                    seq_pdb = line[1:].strip('\n').split('\t')[1]
                    dict_seq[pid_chain_uid] = seq_pdb
    return dict_seq

def get_label(func):
    if func == 'mf':
        mf_dict = {}
        mf_func = []
        mf_label_dict = {}
        with open('/media/administrator/SEAGATE-4T/public/PFP/Struct2Go/data_collect/mf/external_GO_label.txt', 'r') as f:
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
        with open('/media/administrator/SEAGATE-4T/public/PFP/Struct2Go/data_collect/bp/external_GO_label.txt', 'r') as f:
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
        with open('/media/administrator/SEAGATE-4T/public/PFP/Struct2Go/data_collect/cc/external_GO_label.txt', 'r') as f:
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



def loading_contact_map(data_source,threshold):
    contact_map_list = os.listdir('./data_collect/contact_map_dense_pdb_8.0_for_external_test')
    dict_contact_map = {}
    for item in tqdm(contact_map_list):
        if data_source == 'pdb':
            pid_chain_uid = item.strip().split('.')[0]
            dict_contact_map[pid_chain_uid] = np.load('./data_collect/contact_map_dense_pdb_8.0_for_external_test/'+item)
    return dict_contact_map

def external_test(args):
    warnings.filterwarnings("ignore")
    device = torch.device(args.device)
    model_feature = ProSEMT.load_pretrained().to(device).eval()

    print('loading contact maps for external pdb structures...')
    dict_contact_pdb = loading_contact_map('pdb',args.threshold)#over
    print('loading seq for external pdb structures...')
    dict_seq_pdb = loading_seq('pdb')#over

    for func in args.func:
        print('############################################################')
        print(func)
        label,label_num = get_label(func)#mf over
        external_test_item = loading_item_data(func,'external_test')#over

        for data_source in args.data:


            external_test_data = PFPDataset(train_data_X=external_test_item, train_contactmap_X=dict_contact_pdb,train_feature_matrix_X=dict_seq_pdb, train_data_Y=label,model_feature=model_feature, device=device)

            dataset_test = DataLoader(external_test_data, batch_size=64, shuffle=False, collate_fn=collate, drop_last=False)
            model = Model_Net(output_dim=label_num, net=args.net, hidden_dim=args.hidden_dim, pool=args.pool,dropout=args.dropout).to(device)
            model_path = './data_collect/'+func+'/model/'+data_source+'_'+func+'_'+args.net+'_'+str(args.hidden_dim)+'_'+args.pool+'_'+str(args.dropout)+'_'+str(args.threshold)+'.pkl'

            params_dict = torch.load(model_path)
            params_dict["prot_conv1.lin.weight"] = params_dict["prot_conv1.weight"].t()
            del params_dict["prot_conv1.weight"]
            params_dict["prot_conv2.lin.weight"] = params_dict["prot_conv2.weight"].t()
            del params_dict["prot_conv2.weight"]
            params_dict["prot_conv3.lin.weight"] = params_dict["prot_conv3.weight"].t()
            del params_dict["prot_conv3.weight"]
            
            model.load_state_dict(params_dict)
            model.eval()
            bceloss = nn.BCELoss()
            total_preds = torch.Tensor()
            total_labels = torch.Tensor()
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataset_test):
                    data_prot = batch.to(device)
                    output_test = model(data_prot)
                    total_preds = torch.cat((total_preds, output_test.cpu()), 0)
                    total_labels = torch.cat((total_labels, data_prot.label.cpu()), 0)
                loss_test = bceloss(total_preds, total_labels)
            perf = get_results(total_labels.cpu().numpy(), total_preds.cpu().numpy())
            print( '\tTest loss for '+data_source+':\t',loss_test.numpy(), '\tM-AUPR:\t', perf['all']['M-aupr'], '\tm-AUPR:\t', perf['all']['m-aupr'],'\tF-max:\t', perf['all']['F-max'])
        #break



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--func', type=lambda s: [item for item in s.split(",")],
                        default=['mf', 'bp', 'cc'], help="list of func to predict.")
    parser.add_argument('--data', type=lambda s: [item for item in s.split(",")],
                        default=['pdb','alpha'], help="data source.")
    parser.add_argument('--device', type=str, default='cuda:1', help="cuda for model")
    parser.add_argument('--pool', type=str, default='gap-gmp', help="pool for model(gep、gap、gmp、gap-gep、gap-gmp)")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout for model")
    parser.add_argument('--hidden_dim', type=int, default=512, help="hidden dim for linear")
    parser.add_argument('--net', type=str, default='GCN', help="GCN or GAT for model")
    parser.add_argument('--threshold', type=float, default=8.0, help="distance threshold between residues")
    args = parser.parse_args()
    external_test(args)