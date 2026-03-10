clear
clc

% Initial path
main_path = pwd;
data_path = fullfile(main_path,'Data/Preprocessed_EEG');
save_path = fullfile(main_path,'Data/EEG_pure');
mkdir(save_path)
subject_list = fun_searfile(data_path, 'mat');
label_path = fullfile(data_path, subject_list{end});
subject_path = [subject_list(19:45); subject_list(1:18)];
channel_loc = '/Users/xuchengliu/Documents/PhD Researc-h/Research Project/Emotion Recognition/SEED/Emotion-10-20-Cap62.locs';

% Initial parameters
fs = 200;
window_len = 200 * 40;
check_step = 2; % number of the points around the outliers to remove
n_chan = 62;

% Start process
for turn = 1:size(subject_path, 1)
    
    subject = ceil(turn / 3);
    day = turn - (subject - 1) * 3;
    if day == 1
        data_pure = cell(3, 15);
    end
    turn_path = subject_path{turn};
    trial_list = who('-file', fullfile(data_path, turn_path));
    trial_list = [trial_list(1); trial_list(8:15); trial_list(2:7)];
    for trial = 1:size(trial_list, 1)
        
        % Report progress
        report_info = 'Subject: %d, Day: %d, Trial: %d\n';
        fprintf(report_info, subject, day, trial);
        
        % Load data
        data_raw = load(fullfile(data_path, turn_path), trial_list{trial});
        data_raw = data_raw.(trial_list{trial});
        
        % Correct bad channels
        if turn == 1
            data_raw(56, :) = mean(data_raw([48, 55, 57, 62], :), 1);
        elseif turn == 2
            data_raw(45, :) = mean(data_raw([36, 44, 53, 46], :), 1);
        elseif turn == 13
            data_raw(42, :) = mean(data_raw([33, 43, 51], :), 1);
        elseif turn == 16
            data_raw(5, :) = mean(data_raw([3, 11, 12, 13], :), 1);
        elseif turn == 37
            data_raw(54, :) = mean(data_raw([46, 53, 55, 59], :), 1);
        elseif turn == 43
            data_raw(16, :) = mean(data_raw([15, 25, 17, 7], :), 1);
            data_raw(19, :) = mean(data_raw([10, 18, 28, 20], :), 1);
            data_raw(27, :) = mean(data_raw([26, 36, 28, 18], :), 1);
            if trial == 2
                data_raw(62, :) = mean(data_raw([60, 56, 57], :), 1);
            end
        elseif turn == 45 && trial > 11
            data_raw(10, :) = mean(data_raw([9, 19, 11], :), 1);
        end

        % Rereference
        data_raw = reref(data_raw);
        % Filter
        Wn = [0.25*2 50*2] / 200;
        [k,l] = butter(2, Wn);
        data_raw = filtfilt(k, l, data_raw')';
        % Remove baseline
        data_raw = detrend(data_raw')';
        % Remove outliers
        ind_rm = fun_rmout(data_raw, check_step, 130);
        data_raw(:, ind_rm) = [];
        
        % Initial ICA process
        seg_num = floor(size(data_raw, 2) / window_len);
        data_collect = zeros(n_chan, seg_num * window_len);
        % Perform ICA to remove eye artifact
        for seg = 1:seg_num
            % Extract data
            data_seg = data_raw(:, (seg - 1) * window_len + 1: seg * window_len);
            data_reject = fun_rmeye_test(data_seg);
            data_collect(:, (seg - 1) * window_len + 1: seg * window_len) = data_reject;
%             figure
%             plot(data_seg(1, :))
%             hold on
%             plot(data_reject(1, :))
        end
        
        % Store pure data
        data_pure{day, trial} = data_collect;
        
    end
    
    % Save data
    if day == 3
        save(fullfile(save_path, ['S', num2str(subject)]), "data_pure")
    end
end