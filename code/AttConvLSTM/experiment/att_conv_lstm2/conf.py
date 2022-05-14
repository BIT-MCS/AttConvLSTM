CONF = {
    'DATA_WIDTH': 32,
    'DATA_CHANNEL': 2,
    'BATCH': 32,  # bj->32  didi->16  rm->32

    'SEQ_START': int(4),
    'SEQ_LENGTH': int(9),
    'TEST_SEQ_LENGTH': 9,
    # 'SEQ_START': int(4),
    # 'SEQ_LENGTH': int(5),
    # 'TEST_SEQ_LENGTH': 5,

    # for show example
    'example_name': 'pred_5',
    'example_num': 10,
    # for show example

    # bj->10,1  didi->(0.05,1)
    'GAMMA': 0.03,  # rome->0.001,1
    'BETA': 1,

    'STEP_START': int(0),
    'MAX_STEP': int(500000),

    'KEEP_PROB': 0.4,
    'LR': .001,
    'IS_BN': True,
    'DECAY_STEPS': 1000,  # 800-1000
    'DECAY_RATE': 0.995,  # 0.99-0.995

    'IS_SINGLE': False,
    'SAVE_MODEL': False,
    'LOAD_MODEL': True,
    'IS_TEST': True,
    'PATH': '/home/linc/two_storage/PCZ_convlstm_flow_final/convlstm_flow_new/experiment/att_conv_lstm2/bj_32_24_5frame',
    # 'PATH': '/home/linc/two_storage/PCZ_convlstm_flow_final/convlstm_flow_new/experiment/att_conv_lstm2/bj_32_24_1frame',
    # 'PATH':'/home/linc/two_storage/PCZ_convlstm_flow_final/convlstm_flow_new/experiment/att_conv_lstm2/rm_32_5_5frame',
    # 'PATH':'/home/linc/two_storage/PCZ_convlstm_flow_final/convlstm_flow_new/experiment/att_conv_lstm2/didi_16_5_5frame',#  0.05  2->0.08  3->0.1  4->0.03

    # 'TRUTH_RESULT_PATH': '/home/linc/Desktop/convlstm_flow_new/result/AttConvLSTM2/didi_AttConvLSTM2_1_truth.npy',
    # 'PREDICT_RESULT_PATH': '/home/linc/Desktop/convlstm_flow_new/result/AttConvLSTM2/didi_AttConvLSTM2_1_predict.npy',
    # 'TRUTH_RESULT_PATH':'/home/linc/Desktop/convlstm_flow_new/result/AttConvLSTM2/rome_AttConvLSTM2_1_truth.npy',
    # 'PREDICT_RESULT_PATH':'/home/linc/Desktop/convlstm_flow_new/result/AttConvLSTM2/rome_AttConvLSTM2_1_predict.npy',
    'SAVE_DELTA': 1000,
    'CNN_LAYERS': 24,
    # 'CNN_LAYERS': 5,

}

