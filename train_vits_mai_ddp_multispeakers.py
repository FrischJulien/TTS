import os
from glob import glob

from TTS.config.shared_configs import BaseAudioConfig
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname(os.path.abspath(__file__))


dataset_config = [
    BaseDatasetConfig(name="mailabs", meta_file_train=None, path="../datasets/mailabs/fr_FR/", language="fr_FR")
]

audio_config = BaseAudioConfig(
    sample_rate=16000,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    preemphasis=0.0,
    ref_level_db=20,
    log_func="np.log",
    #########################
    do_trim_silence=True,
    ########################
    trim_db=23.0,
    mel_fmin=0,
    mel_fmax=None,
    spec_gain=1.0,
    signal_norm=True,
    do_amp_to_db_linear=False,
    resample=False,
)


#######################
vitsArgs = VitsArgs(
    use_language_embedding=False,
    #embedded_language_dim=4,
    use_speaker_embedding=True,
    use_sdp=False,
)
###########################


config = VitsConfig(
    model_args=vitsArgs,
    audio=audio_config,
    run_name="vits_mai_multispeakers2_french",
    ####################################
    use_speaker_embedding=True,
    ##################################
    batch_size=10*4,
    eval_batch_size=6,
    batch_group_size=0,
    num_loader_workers=2*4,
    num_eval_loader_workers=2*4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="french_cleaners",
    use_phonemes=False,
    phoneme_language="fr-fr",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    
    ##############
    use_language_weighted_sampler=False,
    #####################
    
    print_eval=False,
    
    ####################
    save_step = 2000,
    ####################
    
    ######################
    mixed_precision=True,
    ######################
    
    sort_by_audio_len=True,
    min_seq_len=32 * 256 * 4,
    max_seq_len=160000,
    output_path=output_path,
    datasets=dataset_config,
    characters={
        "pad": "_",
        "eos": "&",
        "bos": "*",
        "characters": "!¡'(),-.:;¿?abcdefghijklmnopqrstuvwxyzµßàáâäåæçèéêëìíîïñòóôöùúûüąćęłńœśşźżƒабвгдежзийклмнопрстуфхцчшщъыьэюяёєіїґӧ «°±µ»$%&‘’‚“`”„",
        "punctuations": "!¡'(),-.:;¿? ",
        "phonemes": None,
        "unique": True,
    },
    test_sentences=[
        [
            "Il m'a fallu beaucoup de temps pour d\u00e9velopper une voix, et maintenant que je l'ai, je ne vais pas me taire.",
            "ezwa",
            None,
            "fr_FR",
        ],
    ],
)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)

##########################################################
# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_speaker_ids_from_data(train_samples + eval_samples)
config.model_args.num_speakers = speaker_manager.num_speakers
###########################################################

# init model
model = Vits(config,speaker_manager,)

# init the trainer and 🚀
trainer = Trainer(
    TrainingArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    training_assets={"audio_processor": ap},
)
trainer.fit()
