import pandas as pd
import baselines_new as b
import re

def clean_text(report):
    # text = re.sub(r"[-()\"#/@;:<>{}=~|.?,]", "", text)
    # text = text.replace('[', '').replace(']', '').strip(' ').strip("u'")


    report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
        .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
        .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                    replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' '.join(tokens) #+ ' .'
    return report

    # return text.lower()

def read_data(path='./records/'):
    norm = pd.read_csv(path+'iu_xray_reports_0.csv')
    abnorm = pd.read_csv(path+'iu_xray_reports_01.csv')
    return norm, abnorm

norm, abnorm = read_data()

data = pd.concat([norm,abnorm])

data = norm

cleaned_res = [clean_text(r) for r in data['res']]

cleaned_gts = [clean_text(r) for r in data['gts']]

print(b.all_scores(cleaned_gts, cleaned_res))