import os, torch, numpy as np, cv2, random, glob, python_speech_features, pandas
from scipy.io import wavfile


def generate_audio_set(data_path: str, batch: list) -> dict:
    """
    load the audios in this batch
    """
    audio_set = {}
    for sample in batch:
        data = sample.split('\t')
        video_name = data[0][:13] # 13 for AVDIAR
        data_name = data[0]
        _, audio = wavfile.read(os.path.join(data_path, video_name, data_name + '.wav'))
        audio_set[data_name] = audio
    return audio_set


def overlap(data_name, audio, audio_set):
    """
    Implements the audio augmentation described in the TalkNet paper.
    An audio track from another vieo in the same batch is randomly selected as noise.
    """
    noise_name = random.sample(list(set(list(audio_set.keys())) - {data_name}), 1)[0] # after python 3.10, random.sample no longer automatically converts sets to lists
    noise_audio = audio_set[noise_name]
    snr = [random.uniform(-5, 5)]

    # align noise and original audio to have the same length
    if len(noise_audio) < len(audio):
        shortage = len(audio) - len(noise_audio)
        noise_audio = np.pad(noise_audio, (0, shortage), 'wrap')
    else:
        noise_audio = noise_audio[:len(audio)]
    
    noise_DB = 10 * np.log10(np.mean(abs(noise_audio ** 2)) + 1e-4)
    clean_DB = 10 * np.log10(np.mean(abs(audio ** 2)) + 1e-4)
    noise_audio = np.sqrt(10 ** ((clean_DB - noise_DB - snr) / 10)) * noise_audio
    audio = audio + noise_audio    
    return audio.astype(np.int16)


def load_audio(data: list[str], data_path: str, num_frames: int, augment_audio: bool, batch: list):
    """
    load audio in a batch and optionally apply augments
    """
    data_name: str = data[0]
    fps: float = float(data[2])

    audio_set = generate_audio_set(data_path, batch)
    audio = audio_set[data_name]

    if augment_audio:
        aug_type: int = random.randint(0, 1)
        if aug_type == 1:
            audio = overlap(data_name, audio, audio_set)
        else:
            audio = audio
    
    # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
    audio = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025 * 25 / fps, winstep = 0.010 * 25 / fps)
    max_audio = int(num_frames * 4)
    if audio.shape[0] < max_audio:
        shortage = max_audio - audio.shape[0]
        audio = np.pad(audio, ((0, shortage), (0,0)), 'wrap')
    audio = audio[:int(round(num_frames * 4)),:]  
    return audio


def load_visual(data: list[str], data_path: str, num_frames: int, augment_visual: bool):
    data_name = data[0]
    video_name = data[0][:13] # 13 for AVDIAR
    face_folder_path = os.path.join(data_path, video_name, data_name)
    face_files = glob.glob("%s/*.jpg"%face_folder_path)
    sorted_face_files = sorted(face_files, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    
    faces = []
    img_size: int = 112

    if augment_visual:
        aug_type = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    else:
        aug_type = 'orig'

    for face_file in sorted_face_files[:num_frames]:
        # read face from file transform to grayscale and size of 112 x 112
        face = cv2.imread(face_file)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (img_size, img_size))

        # augment types
        if aug_type == 'orig':
            faces.append(face)
        elif aug_type == 'flip':
            faces.append(cv2.flip(face, 1))
        elif aug_type == 'crop':
            new = int(img_size * random.uniform(0.7, 1))
            x, y = np.random.randint(0, img_size - new), np.random.randint(0, img_size - new)
            faces.append(cv2.resize(face[y:y + new, x:x + new] , (img_size, img_size))) # type: ignore
        elif aug_type == 'rotate':
            M = cv2.getRotationMatrix2D((img_size / 2, img_size / 2), random.uniform(-15, 15), 1)
            faces.append(cv2.warpAffine(face, M, (img_size, img_size)))

    faces = np.array(faces)
    return faces


def load_label(data: list[str], num_frames: int):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    res = np.array(res[:num_frames])
    return res


def load_context_speakers(data: pandas.DataFrame, target_speaker: str):
    """
    Loads context speakers in a video given a target speaker

    data: DataFrame of *_orig.csv
    target_speaker: the speaker ASD is being conducted on; is represented by the speaker's entity id
    """

    video_id = target_speaker[:13] # video name is contained in the first 13 characters of an entity id
    video_data = data[data['video_id'] == video_id] # filter the df to contain only entries from video

    # first and last appearance of the target speaker described by frame_timestamp
    ts_first_appearance = float(video_data[video_data['entity_id'] == target_speaker].iloc[0]['frame_timestamp'])
    ts_last_appearance = float(video_data[video_data['entity_id'] == target_speaker].iloc[-1]['frame_timestamp'])
   
    # apply set() to remove duplicate entries and remove the target speaker
    other_entities = set(video_data['entity_id']) - {target_speaker}

    # remove entities which intersect for less half of the time
    candidate_context_speakers = list()
    for entity in other_entities:
        first_appearance = float(video_data[video_data['entity_id'] == entity].iloc[0]['frame_timestamp'])
        last_appearance = float(video_data[video_data['entity_id'] == entity].iloc[-1]['frame_timestamp'])

        overlap_duration = max(0, min(last_appearance, ts_last_appearance) - max(first_appearance, ts_first_appearance))
        if overlap_duration > (ts_last_appearance - ts_first_appearance) / 2:
            candidate_context_speakers.append([entity, first_appearance, last_appearance])

    return candidate_context_speakers


class TrainLoader():
    def __init__(self, trial_file_name: str, audio_path: str, visual_path: str, batch_size: int):
        self.audio_path  = audio_path
        self.visual_path = visual_path
        self.batches = open(trial_file_name).read().splitlines() # batch size of 1
        self.data = pandas.read_csv('AVDIAR_ASD/csv/train_orig.csv')

        """
        samples = open(trial_file_name).read().splitlines()
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        sorted_samples = sorted(samples, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)         

        # assign samples to batches
        start = 0
        while start != len(sorted_samples):
            length = int(sorted_samples[start].split('\t')[1])
            end = min(len(sorted_samples), start + max(int(batch_size / length), 1))
            self.batches.append(sorted_samples[start:end])
            if end == len(sorted_samples):
                break
            start = end

        for batch in self.batches:
            print(len(batch))
        """

    def __getitem__(self, index):
        """
        returns audio features, visual features, and labels as tuple of Tensors for a batch.
        """

        batch = [self.batches[index]]
        num_frames = int(batch[0].split('\t')[1])
        data: list[str] = batch[0].split('\t')

        audio_features = [load_audio(data, self.audio_path, num_frames, False, batch)]
        visual_features = [load_visual(data, self.visual_path, num_frames, False)]
        labels = [load_label(data, num_frames)]
        context_speaker_features = []

        data = batch[0].split('\t')
        target_speaker = data[0]
        for context_speaker, first_appearance, _ in load_context_speakers(self.data, data[0]):
            unaligned: np.ndarray = load_visual([context_speaker], self.visual_path, num_frames, True)
                
            # first appearance of the target speaker described by frame_timestamp
            ts_first_appearance = float(self.data[self.data['entity_id'] == target_speaker].iloc[0]['frame_timestamp'])

            # align visual feature of context speaker with features of target speaker temporally by zero-padding
            # calculate the amount of frames between first appearances
            gap = int(abs(first_appearance - ts_first_appearance) / 0.04)
            if ts_first_appearance < first_appearance:
                # zero pad the beginning
                unaligned = np.concatenate((np.zeros((gap, 112, 112)), unaligned))
            else:
                unaligned = unaligned[gap:]

            if len(unaligned) > num_frames:
                aligned = unaligned[:num_frames]
            else:
                gap = num_frames - len(unaligned)
                aligned = np.concatenate((unaligned, np.zeros((gap, 112, 112))))
                
            context_speaker_features.append(aligned)

        return (torch.FloatTensor(np.array(audio_features)),
                torch.FloatTensor(np.array(visual_features)),
                torch.FloatTensor(np.array(context_speaker_features)),
                torch.LongTensor(np.array(labels)))

    def __len__(self):
        """
        returns the number of batches
        """
        return len(self.batches)


class ValLoader():
    def __init__(self, trial_file_name, audio_path, visual_path):
        self.audio_path = audio_path
        self.visual_path = visual_path
        self.batches = open(trial_file_name).read().splitlines() # batch size of 1
        self.data = pandas.read_csv('AVDIAR_ASD/csv/val_orig.csv')

    def __getitem__(self, index):
        """
        returns audio features, visual features, and labels as tuple of Tensors for a "batch".

        batch size for validation set is 1.
        variable names are in terms of batches to be consistent with TrainLoader and other functions written for it
        """

        batch = [self.batches[index]]
        num_frames = int(batch[0].split('\t')[1])
        data: list[str] = batch[0].split('\t')

        audio_features = [load_audio(data, self.audio_path, num_frames, False, batch)]
        visual_features = [load_visual(data, self.visual_path, num_frames, False)]
        labels = [load_label(data, num_frames)]
        context_speaker_features = []

        data = batch[0].split('\t')
        target_speaker = data[0]
        for context_speaker, first_appearance, _ in load_context_speakers(self.data, data[0]):
            unaligned: np.ndarray = load_visual([context_speaker], self.visual_path, num_frames, True)
                
            # first appearance of the target speaker described by frame_timestamp
            ts_first_appearance = float(self.data[self.data['entity_id'] == target_speaker].iloc[0]['frame_timestamp'])

            # align visual feature of context speaker with features of target speaker temporally by zero-padding
            # calculate the amount of frames between first appearances
            gap = int(abs(first_appearance - ts_first_appearance) / 0.04)
            if ts_first_appearance < first_appearance:
                # zero pad the beginning
                unaligned = np.concatenate((np.zeros((gap, 112, 112)), unaligned))
            else:
                unaligned = unaligned[gap:]

            if len(unaligned) > num_frames:
                aligned = unaligned[:num_frames]
            else:
                gap = num_frames - len(unaligned)
                aligned = np.concatenate((unaligned, np.zeros((gap, 112, 112))))
                
            context_speaker_features.append(aligned)

        return (torch.FloatTensor(np.array(audio_features)),
                torch.FloatTensor(np.array(visual_features)),
                torch.FloatTensor(np.array(context_speaker_features)),
                torch.LongTensor(np.array(labels)))

    def __len__(self):
        """
        the batch size for validation set is 1 
        so this function just returns the number of total samples
        """
        return len(self.batches)
