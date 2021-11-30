from model import Model_Net
import argparse,os,warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import loading_contact_map,loading_seq,get_label,loading_item_data,PFPDataset,collate,get_results
from featureGen import ProSEMT

def test(args):
    warnings.filterwarnings("ignore")
    device = torch.device(args.device)
    model_feature = ProSEMT.load_pretrained().to(device).eval()

    print('loading contact maps for pdb structures...')
    dict_contact_pdb = loading_contact_map('pdb',args.threshold)
    print('loading seq for pdb structures...')
    dict_seq_pdb = loading_seq('pdb')
    for func in args.func:
        print('############################################################')
        print(func)
        label,label_num = get_label(func)#mf
        test_item = loading_item_data(func,'test')
        for data_source in args.data:


            test_data = PFPDataset(train_data_X=test_item, train_contactmap_X=dict_contact_pdb,train_feature_matrix_X=dict_seq_pdb, train_data_Y=label,model_feature=model_feature, device=device)

            dataset_test = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate, drop_last=False)
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
    test(args)