from nltk.tag.stanford import StanfordPOSTagger

st = StanfordPOSTagger('stanford-postagger/models/english-bidirectional-distsim.tagger',
                       'stanford-postagger/stanford-postagger.jar')

print(st.tag('while preparing for battle I always found keys'.split()))

st2 = StanfordPOSTagger('stanford-postagger/models/english-bidirectional-distsim.tagger',
                        'stanford-postagger/stanford-postagger-4.2.0.jar')
print(st2.tag('while preparing for battle I always found keys'.split()))
