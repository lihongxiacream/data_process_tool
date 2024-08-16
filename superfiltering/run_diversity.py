from diversity import Superfiltering_Diversity
import argparse

def main(args):
    instance = Superfiltering_Diversity(data_path=args.data_path,          # 原始数据路径
                                      embed_model_path=args.embed_model_path,
                                      emb_type=args.emb_type,              # ['whole', 'instruction', 'response']
                                      json_save_path=args.json_save_path,
                                      diversity=args.fla_num,          # 多样性只适用于单轮对话
                                      batch_size=args.batch_size)

    instance.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Superfiltering Diversity Script')

    # 添加命令行参数
    parser.add_argument('--data_path', type=str, required=True, help='原始数据路径')
    parser.add_argument('--json_save_path', type=str, required=True, help='结果保存路径')
    parser.add_argument("--embed_model_path", type=str, required=True,default='code_diversity_fla/all-MiniLM-L6-v2')
    parser.add_argument("--emb_type", type=str, choices=['whole', 'instruction', 'response'],default='instruction', help='whole, instruction, response')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--fla_num', type=int, default=5000, help='多样性筛选数量')

    args = parser.parse_args()
    main(args)