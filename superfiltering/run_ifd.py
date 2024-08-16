from ifd import Superfiltering_IFD
import argparse

def main(args):
    instance = Superfiltering_IFD(data_path=args.data_path,               # 原始数据路径
                                  post_data_path=args.post_data_path,     # 带有IFD且不筛选的数据路径
                                  model_name_or_path=args.model_name_or_path,
                                  sample_rate=args.ifd_rate,              # IFD筛选后数据路径
                                  json_save_path=args.json_save_path,
                                  data_type=args.data_type)

    instance.run()
    # instance.run_only_select() # 只筛选数据

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Superfiltering IFD Script')

    # 添加命令行参数
    parser.add_argument('--data_path', type=str, required=True, help='原始数据路径')
    parser.add_argument('--post_data_path', type=str, required=False, default='outputs/.cache/postprocess_data.jsonl',help='带有IFD的数据路径')
    parser.add_argument('--model_name_or_path', type=str, required=True, default='gpt2/',help='模型名称或路径')
    parser.add_argument('--json_save_path', type=str, required=True, help='结果保存路径')
    parser.add_argument('--data_type', type=str, choices=['Sharegpt', 'Alpaca'], default='Alpaca', required=True, help="只能为'Sharegpt'或'Alpaca'格式")
    parser.add_argument('--ifd_rate', type=float, required=False, help='IFD筛选比例')

    args = parser.parse_args()

    main(args)
