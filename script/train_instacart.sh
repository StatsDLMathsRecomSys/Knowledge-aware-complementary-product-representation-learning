## item word dictionary and user trx history are mandatory
ITEM_WORD_INPUT="../data/instacart/ml_context.tsv"
USER_HIST_INPUT="../data/instacart/ml_history.tsv"
# -itemWordInput ${ITEM_WORD_INPUT} -userHistInput ${USER_HIST_INPUT}

## user word dictionary and their view history etc are optional
USER_WORD_INPUT="../abc"
#-userWordInput ${USER_WORD_INPUT}
USER_VIEW_HIST_INPUT="../abc"
# -userHistInputView ${USER_VIEW_HIST_INPUT}

OUTPUT_PREFIX="../output/insta_free"
NUM_THREAD=6

NUM_EPOCHS=20
DIM=100
USER_DIM=30
NEG=5

# add -skipContext flag to skip conextual constraints for item embeddings
# add -skipUserContext flag to skip contextual constraints for user embeddings
# add -regOutput for fast computation at the cost of getting weaker regularization.

../build/uni-vec train -itemWordInput ${ITEM_WORD_INPUT} \
-userHistInput ${USER_HIST_INPUT}  \
-output ${OUTPUT_PREFIX} -thread ${NUM_THREAD} -epoch ${NUM_EPOCHS} \
-dim ${DIM} -userDim ${USER_DIM} -neg ${NEG} -skipUserContext -skipContext #-regOutput  -skipUserContext

