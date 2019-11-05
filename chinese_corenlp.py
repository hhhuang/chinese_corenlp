from collections import deque
import json
import urllib.request
import urllib.parse

from hanziconv.hanziconv import HanziConv

class ChineseCoreNLP(object):
    """
        Run the CoreNLP server for Chinese processing as following command:
        nohup java -Xmx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties chinese.properties -port 9000 -timeout 15000 &
        The chinese.properties file should be given for setting.
        More information: https://stanfordnlp.github.io/CoreNLP/corenlp-server.html

        The expected input of the Chinese model provided by Stanford CoreNLP is in Simplified Chinese. For the input in Traditional Chinese, the simplification is performed at the first. Then, the original Traditional tokens will be restored to the output of CoreNLP. 
    """
    def __init__(self, host="localhost", port=9000, traditional=True):
        self.url = "http://%s:%d" % (host, port)
        self.traditional = traditional
    
    def build_tree(self, text, tokens):
        text = text.strip(" ")
        if text[0] != '(' or text[-1] != ')':
            raise ValueError('The format of the parse tree is invalid: %s.' % text)
        text = text[1:-1].strip()

        label = ""
        for i in range(len(text)):
            if text[i] == ' ':
                break
            else:
                label += text[i]
        node = {'label': label, 'children': [], 'token': None}
        if text[-1] == ')':
            cnt = 0
            p = i
            for i in range(i, len(text)):
                if text[i] == '(':
                    cnt += 1
                elif text[i] == ')':
                    cnt -= 1
                    if cnt == 0:
                        node['children'].append(self.build_tree(text[p:i+1], tokens))
                        p = i + 1
        else:
            if label != tokens[0]['pos']:
                raise ValueError('The POS tags do not match. (%s, %s) (%s, %s)' % (
                    label, text[i:].strip(), tokens[0]['pos'], tokens[0]['word']))
            node['token'] = tokens.popleft()['word']
        return node

    def build_parse_tree(self, text, tokens):
        return self.build_tree(text.replace("\n", " "), tokens)

    def perform(self, text):
        if self.traditional:
            simplified_text = HanziConv.toSimplified(text)
        else:
            simplified_text = text
        try:
            req = urllib.request.Request(url=self.url, 
                data=simplified_text.encode("utf-8"), 
                method="POST")
            with urllib.request.urlopen(req, timeout=15000) as f:
                results = json.loads(f.read().decode("utf-8"))
        except Exception:
            print("Fail to parse the input: %s" % text)
            return []
        return self.output(results, text)

    def build_entitymentions(self, mentions, text): 
        if not self.traditional:
            return mentions
        for mention in mentions:
            mention['text'] = text[mention['characterOffsetBegin']:mention['characterOffsetEnd']]
        return mentions

    def build_dependencies(self, dependencies, tokens):
        if not self.traditional:
            return dependencies
        for dep in dependencies:
            if dep['governor'] > 0:
                dep['governorGloss'] = tokens[dep['governor'] - 1]['word']
            if dep['dependent'] > 0:
                dep['dependentGloss'] = tokens[dep['dependent'] - 1]['word']
        return dependencies

    def build_tokens(self, tokens, text):
        if not self.traditional:
            return tokens
        for tok in tokens:
            tok['word'] = tok['originalText'] = tok['lemma'] = text[tok['characterOffsetBegin']:tok['characterOffsetEnd']]
        return tokens

    def output(self, raw_results, text): 
        results = []
        for sent in raw_results['sentences']:
            sent['tokens'] = self.build_tokens(sent['tokens'], text)
            sent['entitymentions'] = self.build_entitymentions(sent['entitymentions'], text)
            sent['parse'] = self.build_parse_tree(sent['parse'], deque(sent['tokens']))
            for dep_type in ["basicDependencies", "enhancedDependencies", "enhancedPlusPlusDependencies"]:
                sent[dep_type] = self.build_dependencies(sent[dep_type], sent['tokens'])
            results.append(sent)
        return results


if __name__ == "__main__":
    nlp = ChineseCoreNLP()
    print(nlp.perform("我想聽陳曉東的歌。也想看王家衛的電影。明天要去淡水，請幫我找相關資料。"))

