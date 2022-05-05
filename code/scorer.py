import pandas as pd
from tqdm import tqdm
from model.QACGBERT import *
from util.tokenization import *
from torch.utils.data import DataLoader, TensorDataset
import random


context_id_map_fiqa = {"legal": 0, "m&a": 1, "regulatory": 2, "risks": 3, "rumors": 4, "company communication": 5,
                       "trade": 6, "central banks": 7, "market": 8, "volatility": 9, "financial": 10,
                       "fundamentals": 11, "price action": 12, "insider activity": 13, "ipo": 14, "others": 15}


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, score, seq_len, context_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.score = score
        self.seq_len = seq_len
        self.context_ids = context_ids


def convert_to_unicode(text):
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python 3")


def get_test_examples(path):
    test_data = pd.read_csv(path).values
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = convert_to_unicode(str(line[4]))
            text_b = convert_to_unicode(str(line[3]))
            label = convert_to_unicode(str(line[2]))
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    return _create_examples(test_data, "test")


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, max_seq_length,
                                 tokenizer, max_context_length,
                                 context_standalone, args):
    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        tokens_context = None
        if example.text_b:
            tokens_context = tokenizer.tokenize(example.text_b)

        if tokens_b and not context_standalone:
            truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b and not context_standalone:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        context_ids = []
        if tokens_context:
            context_ids = [context_id_map_fiqa[example.text_b]]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        seq_len = len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        while len(context_ids) < max_context_length:
            context_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(context_ids) == max_context_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                score=example.label,
                seq_len=seq_len,
                context_ids=context_ids))

    return features


def get_model_and_tokenizer(vocab_file,
                               bert_config_file=None, init_checkpoint=None,
                               do_lower_case=True,
                               init_lrp=False):
    tokenizer = FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case, pretrain=False)
    if bert_config_file is not None:
        bert_config = BertConfig.from_json_file(bert_config_file)
    else:
        bert_config = BertConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02
        )
    bert_config.vocab_size = len(tokenizer.vocab)
    model = QACGBertForSequenceScore(
                bert_config,
                init_weight=True,
                init_lrp=init_lrp)

    if init_checkpoint is not None:
        if "checkpoint" in init_checkpoint:
            state_dict = torch.load(init_checkpoint, map_location='cpu')
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    name = k[7:]
                    new_state_dict[name] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        else:
            model.bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'), strict=False)
    return model, tokenizer


def system_setups(args):
    # system related setups
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.bert_config_file is not None:
        bert_config = BertConfig.from_json_file(args.bert_config_file)
        if args.max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                args.max_seq_length, bert_config.max_position_embeddings))

    return device, n_gpu


def data_and_model_loader(device, n_gpu, args):
    model, tokenizer = get_model_and_tokenizer(vocab_file=args.vocab_file,
                                bert_config_file=args.bert_config_file, init_checkpoint=args.init_checkpoint,
                                do_lower_case=True,
                                init_lrp=False)

    test_examples = get_test_examples(args.path)
    test_features = convert_examples_to_features(
        test_examples, args.max_seq_length,
        tokenizer, args.max_context_length,
        args.context_standalone, args)

    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    all_seq_len = torch.tensor([[f.seq_len] for f in test_features], dtype=torch.long)
    all_context_ids = torch.tensor([f.context_ids for f in test_features], dtype=torch.long)

    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_seq_len, all_context_ids)
    test_dataloader = DataLoader(test_data, shuffle=False)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    return model, test_dataloader


def pred(args):
    device, n_gpu = system_setups(args)
    model, test_dataloader = data_and_model_loader(device, n_gpu, args)

    model.eval()
    pbar = tqdm(test_dataloader, desc="Iteration")
    score = []
    with torch.no_grad():
        for _, batch in enumerate(pbar):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            input_ids, input_mask, segment_ids, score, seq_lens, \
                context_ids = batch
            max_seq_lens = max(seq_lens)[0]
            input_ids = input_ids[:,:max_seq_lens]
            input_mask = input_mask[:,:max_seq_lens]
            segment_ids = segment_ids[:,:max_seq_lens]

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            score = score.to(device)
            seq_lens = seq_lens.to(device)
            context_ids = context_ids.to(device)

    return score


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--max_seq_length")
    parser.add_argument("--vocab_file")
    parser.add_argument("--bert_config_file")
    parser.add_argument("--init_checkpoint")
    parser.add_argument("--seed")
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--no_cuda", default=False, action='store_true')
    parser.add_argument("--max_context_length", default=1, type=int)
    parser.add_argument("--context_standalone", default=False, action='store_true')
    args = parser.parse_args()

    score = pred(args)
    print(score)
