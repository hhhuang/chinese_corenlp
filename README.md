# Chinese CoreNLP
A Python wrapper of [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/index.html) for Chinese Processing. 

## Features
* Based on the powerful Stanford CoreNLP toolkit (Manning et al., 2014), a number of tasks in Chinese processing will be performed for the given text, including sentence segmentation, Chinese word segmentation (中文斷詞), part-of-speech tagging (詞性標記), constituency (syntactic) parsing, dependency parsing, named entity recognition, and mention detection. 
* The input can be in either Traditional (default) or Simplfied Chinese. Originally, the expected input of the Stanford CoreNLP models is in Simplified Chinese. For the input in Traditional Chinese, this wrapper will convert the input into Simplified Chinese (with the HanziConv package), and restore the original tokens in the outcomes of the models. 

## Usage
To use this package, you have to run the [Stanford CoreNLP toolkit](https://stanfordnlp.github.io/CoreNLP/corenlp-server.html) as a server with the settings in the file [chinese.properties](https://github.com/hhhuang/chinese_corenlp/blob/master/chinese.properties). 
  
  ```
  nohup java -Xmx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties chinese.properties -port 9000 -timeout 15000 &
  ```
  
Use the wrapper as the following example. 

  ```
  from chinese_corenlp import ChineseCoreNLP
  
  nlp = ChineseCoreNLP()
  results = nlp.perform("我想聽陳曉東的歌。也想看王家衛的電影。明天要去淡水，請幫我找相關資料。")
  ```
  
The outcome consists of three sentences with following annotations: 

* tokens: The outcomes of Chinese word segmentation and part-of-speech tagging. 
* parse: The constituency parse tree of the sentence. 
* basicDependencies, enhancedDependencies, enhancedPlusPlusDependencies: The outcomes of depdendency parsing (three variations).
* entitymentions: The named entities in the sentence. 

## Arguments
The default input is in Traditional Chinese. Switch to Simplified Chinese by the argument as follows.

  ```
  nlp = ChineseCoreNLP(traditional=False)
  ```

So far, additional arguments are host ("localhost" by default) and port (9000 by default) of the CoreNLP server.  
