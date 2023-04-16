import os
import pickle
from bs4 import BeautifulSoup as BSHTML
from requests.models import JSONDecodeError
from langchain.text_splitter import CharacterTextSplitter


def preprocess_and_pickle(page_iter, src_name):
    docs = []
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1024)
    for page in page_iter:
        docs.extend((splitter.create_documents([page['text']], [page['metadata']])))
    pickle.dump(docs, open(f'knowledgebase/{src_name}.pkl', 'wb'))


def scrape_blogs():
    with tempfile.TemporaryDirectory() as d:

        # clone
        subprocess.check_call(
            f"git clone --depth 1 https://github.com/pytorch/pytorch.github.io.git .",
            cwd=d,
            shell=True,
        )
        repo_path = pathlib.Path(d)

        # filter files
        markdown_files = list(repo_path.glob("_posts/*.md"))

        # process
        for markdown_file in markdown_files:
            filename = markdown_file.parts[-1]
            title = os.path.splitext('-'.join(filename.split('-')[3:]))[0]
            blog_url = f"https://pytorch.org/blog/{title}/"
            with open(markdown_file, "r") as f:
                yield {'text': f.read(), 'metadata': {"source": blog_url}}


def get_forum(period='weekly'):
    host = "https://discuss.pytorch.org"

    def _get_accepted_topics(period, page=0, dst=[]):
        resp = requests.get(host+f'/top.json?page={page}&period={period}&per_page=100').json()
        dst.extend([(d['id'], d['title']) for d in resp['topic_list']['topics'] if d['has_accepted_answer'] is True])
        if 'more_topics_url' in resp['topic_list'].keys():
            page += 1
            _get_accepted_topics(period=period, page=page, dst=dst)
        return dst

    def _process_cooked(cooked):
        bs = BSHTML(cooked)
        p = ' '.join([x.get_text() for x in bs.find_all('p')])
        return p

    solved_topics = _get_accepted_topics(period)
    for t, title in solved_topics:
        try:
            r = requests.get(host+f'/t/{t}/posts.json').json()
        except JSONDecodeError:
            continue
        try:
            q = title + '? ' + _process_cooked(r['post_stream']['posts'][0]['cooked'])
            a = _process_cooked([x['cooked'] for x in r['post_stream']['posts'] if x['accepted_answer'] is True][0])
        except IndexError:
            print(f"Skipping https://discuss.pytorch.org/t/{t}/")
            continue
        text = "QUESTION: " + q + ' ANSWER: ' + a
        yield {'text': text, 'metadata': {'source': f"https://discuss.pytorch.org/t/{t}/"}}


def get_docs():
    with tempfile.TemporaryDirectory() as d:
        subprocess.check_call(
            f"git clone --depth 1 https://github.com/pytorch/pytorch.git .",
            cwd=d,
            shell=True,
        )
        repo_path = pathlib.Path(d + '/docs/source')
        markdown_files = list(repo_path.glob("**/*.rst"))
        for markdown_file in markdown_files:
            relative_path = markdown_file.relative_to(repo_path)
            if '_' in markdown_file.name:
                continue
            with open(markdown_file, "r") as f:
                i = markdown_file.parts.index('source')
                filename = os.path.splitext(relative_path)[0]
                page_url = f"https://pytorch.org/docs/stable/{filename}.html"
                yield {'text': f.read(), 'metadata': {"source": page_url}, "file":relative_path}


if __name__ == "__main__":
    preprocess_and_pickle(scrape_blogs(), 'blogs')
    preprocess_and_pickle(get_forum(), 'forum')
    preprocess_and_pickle(get_docs(), 'docs')