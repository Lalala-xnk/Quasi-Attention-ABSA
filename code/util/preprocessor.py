import json
import pandas as pd


def preprocessor(path, name):
    aspects = set()
    df_id = []
    df_sentiment = []
    df_aspect = []
    df_sentence = []
    with open(path + name, 'r') as f:
        data = json.load(f)
        for idx in data:
            for content in data[idx]['info']:
                aspect = content['aspects'].split('/')[0][2:]
                aspects.add(aspect)

    with open(path + name, 'r') as f:
        data = json.load(f)
        for idx in data:
            tmp = data[idx]['info']
            for aspect in aspects:
                sentiment = 0.0
                cnt = 0
                for content in tmp:
                    if content['aspects'].split('/')[0][2:] == aspect:
                        sentiment += float(content['sentiment_score'])
                        cnt += 1
                df_id.append(idx)
                df_sentiment.append(sentiment / cnt if cnt > 0 else 100)
                df_aspect.append(aspect.lower())
                df_sentence.append(data[idx]['sentence'].lower())
    d = {'id': df_id, 'sentiment': df_sentiment, 'aspect': df_aspect, 'sentence': df_sentence}
    df = pd.DataFrame(data=d)
    df.to_csv(path + '/task1_post_ABSA_train.csv')


if __name__ == '__main__':
    path = '/Users/nengkai/PycharmProjects/ocbc/Quasi-Attention-ABSA/datasets/FIQA/FiQA_ABSA_task1/'
    preprocessor(path, 'task1_post_ABSA_train.json')
