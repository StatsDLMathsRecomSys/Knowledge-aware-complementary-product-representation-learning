## item word dictionary and user trx history are mandatory 
ITEM_WORD_INPUT="../test/resource/fake_item_word.txt"
USER_HIST_INPUT="../test/resource/fake_user_hist.txt"
# -itemWordInput ${ITEM_WORD_INPUT} -userHistInput ${USER_HIST_INPUT}

## user word dictionary and their view history etc are optional
USER_WORD_INPUT="../test/resource/fake_user_word.txt"
#-userWordInput ${USER_WORD_INPUT} 
USER_VIEW_HIST_INPUT="../test/resource/fake_user_hist.txt"
# -userHistInputView ${USER_VIEW_HIST_INPUT}

OUTPUT_PREFIX="../output/test"
NUM_THREAD=4

NUM_EPOCHS=10
DIM=100
USER_DIM=100
NEG=5

# add -skipContext flag to skip conextual constraints for item embeddings
# add -skipUserContext flag to skip contextual constraints for user embeddings
# add -regOutput for fast computation at the cost of getting weaker regularization.

../build/uni-vec train -itemWordInput ${ITEM_WORD_INPUT}  -userWordInput ${USER_WORD_INPUT} \
-userHistInput ${USER_HIST_INPUT}  -userHistInputView ${USER_VIEW_HIST_INPUT} \
-output ${OUTPUT_PREFIX} -thread ${NUM_THREAD} -epoch ${NUM_EPOCHS} \
-dim ${DIM} -userDim ${USER_DIM} -neg ${NEG} #-skipContext #-regOutput  -skipUserContext
