age_groups = {'YA','OA'};
for k = 1:length(age_groups)
    %% loop through age groups
    age = age_groups{k};
    
    %% loop through all the files
    folder = sprintf('../output/correlation_map_analysis_data/%s', age);
    files = dir(fullfile(folder, '*.xlsx')); % list all files in the folder

    for i = 1:length(files)
        filename = files(i).name;

        %% Import csv and save as a table
        t = readtable(sprintf('../output/correlation_map_analysis_data/%s/%s', age, filename));

        %% Create 2 seperate structures
        timeseries_gaze.rate = 2;
        timeseries_gaze.name = 'gaze';
        timeseries_gaze.signal = t.gaze;

        timeseries_narrative.rate = 2;
        timeseries_narrative.name = 'narrative';
        timeseries_narrative.signal = t.narrative;
        %% Combine 2 structures into 1
        timeseries(1) = timeseries_gaze;
        timeseries(2) = timeseries_narrative;
        %% correlation map analysis (Adriano Vilela Barbosa et al., 2012)
        % Create TimeSignal objects from the face and forehead motion signals
        signal_gaze = TimeSignal(timeseries(1).signal,timeseries(1).rate);
        signal_narrative = TimeSignal(timeseries(2).signal,timeseries(2).rate);

        % The signal names
        signal_gaze.name = sprintf('signal_gaze');
        signal_narrative.name = sprintf('signal_narrative');
        % ---------- The instantaneous correlation algorithm ---------- %

        % The correlation map parameters
        filter_type = 'ds';
        eta = 0.3;
        delta = 8;
        plot_1d = true;
        interactive = false;
        plot_inputs_together = false;

        % Plot the correlation map
        [handles,corr_map] = correlation_map_gui(signal_gaze,signal_narrative,[],filter_type,eta,delta,plot_1d,interactive,plot_inputs_together);
        output_file = sprintf('../output/correlation_map/%s/%s', age, filename); 
        disp(output_file)
        xlswrite(output_file,corr_map.corr_map)
        % ---------------------------------------------------------------------------- %
    end
end