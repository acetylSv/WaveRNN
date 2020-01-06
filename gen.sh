#tw='/home/acetylsv/WaveRNN/checkpoints/from_bs/taco_step134K_weights.pyt'
tw='/home/acetylsv/WaveRNN/checkpoints/bs/taco_step270K_weights.pyt'
#tw='/home/acetylsv/WaveRNN/checkpoints/ljspeech_lsa_smooth_attention.tacotron/taco_step350K_weights.pyt'
#tw='/home/acetylsv/WaveRNN/checkpoints/ljspeech_lsa_smooth_attention.tacotron/taco_step118K_weights.pyt'
vw='/home/acetylsv/WaveRNN/checkpoints/ljspeech_mol.wavernn/wave_step450K_weights.pyt'
CUDA_VISIBLE_DEVICES='' python3 gen_tacotron.py \
    --tts_weights $tw -a \
    wavernn \
    --voc_weights $vw 
