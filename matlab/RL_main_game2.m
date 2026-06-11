%% matlab/RL_main_game2.m - four-controller Game2 integration runner
%
% This runner is intentionally separate from RL_main_simple.m. It exports a
% single shared state row and consumes one action row with named columns:
% Qec, Qint, SO4ref, and Qw.
%
% First use is an integration test only:
%   1. In MATLAB: init_bsm2
%   2. In a terminal: python agents/ctrl_game2_coordinator.py --mode hold
%   3. In MATLAB: run RL_main_game2
%
% Optional before running:
%   GAME2_STOP_TIME = 2;    % days, default short integration test
%
% Do not use this runner as thesis performance evidence until a clean
% evaluation protocol and retained result export are added.

%% ===============================
% Project paths
%% ===============================
RL_DIR     = fileparts(fileparts(mfilename('fullpath')));
THESIS_DIR = fileparts(RL_DIR);
UNI_DIR    = fileparts(THESIS_DIR);
BSM2_DIR   = fullfile(THESIS_DIR, 'BSM2_R2019b');
if ~isfolder(BSM2_DIR)
    BSM2_DIR = fullfile(UNI_DIR, 'BSM2_R2019b');
end
if ~isfolder(BSM2_DIR)
    error('[GAME2] BSM2_R2019b not found next to Thesis work or Universidade.');
end

cd(BSM2_DIR);
addpath(BSM2_DIR);

model = 'bsm2_cl';
COMMS = fullfile(RL_DIR, 'comms');
addpath(fullfile(RL_DIR, 'matlab'));

if bdIsLoaded(model)
    try
        simStatus = get_param(model, 'SimulationStatus');
    catch
        simStatus = 'stopped';
    end

    if ~strcmpi(simStatus, 'stopped')
        fprintf('[GAME2] Model %s was %s. Stopping before reload...\n', model, simStatus);
        try
            set_param(model, 'SimulationCommand', 'stop');
        catch stopErr
            warning('[GAME2] Failed to request model stop: %s', stopErr.message);
        end

        tStop = tic;
        while bdIsLoaded(model)
            try
                simStatus = get_param(model, 'SimulationStatus');
            catch
                simStatus = 'stopped';
            end

            if strcmpi(simStatus, 'stopped')
                break;
            end
            if toc(tStop) > 60
                error('[GAME2] Timeout waiting for %s to stop before reload.', model);
            end
            pause(0.2);
        end
    end

    try
        close_system(model, 0);
    catch closeErr
        warning('[GAME2] close_system failed once: %s', closeErr.message);
        pause(0.5);
        close_system(model, 0);
    end
end
load_system(model);

%% ===============================
% Time and communication parameters
%% ===============================
dt = 15 / (24*60);
tol = 1e-6;
TIMEOUT = 120;  % wait up to 120 s for the Python action (margin over ~1.6 s/step);
                % with GAME2_ABORT_ON_TIMEOUT=true a real hang aborts (clean-or-abort),
                % so a completed run is provably free of baseline-injection contamination.
FAST_SAMPLES_PER_DAY = max(2, round(1.0 / dt));
SRT_WINDOW_DAYS = 3.0;
SRT_WINDOW_SAMPLES = max(FAST_SAMPLES_PER_DAY, round(SRT_WINDOW_DAYS / dt));

if ~exist('GAME2_START_TIME', 'var') || isempty(GAME2_START_TIME)
    GAME2_START_TIME = 0.0;
end
if ~exist('GAME2_STOP_TIME', 'var') || isempty(GAME2_STOP_TIME)
    GAME2_STOP_TIME = 2.0;
end
if ~exist('GAME2_ABORT_ON_TIMEOUT', 'var') || isempty(GAME2_ABORT_ON_TIMEOUT)
    GAME2_ABORT_ON_TIMEOUT = true;
end
ABORT_ON_TIMEOUT = GAME2_ABORT_ON_TIMEOUT;

FLAG_STATE  = fullfile(COMMS, 'flag_state.run');
FLAG_ACTION = fullfile(COMMS, 'flag_action.run');
FLAG_ERROR  = fullfile(COMMS, 'flag_error.run');
STATE_FILE  = fullfile(COMMS, 'state.csv');
ACTION_FILE = fullfile(COMMS, 'action.csv');

disp('[GAME2] Clearing old communication flags...');
if isfile(FLAG_STATE),  delete(FLAG_STATE);  end
if isfile(FLAG_ACTION), delete(FLAG_ACTION); end
if isfile(FLAG_ERROR),  delete(FLAG_ERROR);  end

%% ===============================
% Configure simulation
%% ===============================
set_param(model, 'SimulationMode', 'accelerator');
set_param(model, 'StartTime', num2str(GAME2_START_TIME));
set_param(model, 'StopTime', num2str(GAME2_STOP_TIME));
set_param(model, 'OutputTimes', sprintf('[%g:(1/96):%g]', GAME2_START_TIME, GAME2_STOP_TIME));
% PauseTimes intentionally not configured: the BSM2 model under MATLAB R2025b
% does not expose this parameter, and stepping is driven manually below via
% SimulationCommand 'pause'/'continue'. Setting it was a no-op (empty value).

set_param(model, 'SimulationCommand', 'start');
fprintf('[GAME2] Simulation started (t=%g -> %g days, four-agent mode)...\n', GAME2_START_TIME, GAME2_STOP_TIME);

pause(0.5);
while strcmp(get_param(model, 'SimulationStatus'), 'initializing')
    pause(0.1);
end

next_pause_time = GAME2_START_TIME + dt;

%% ===============================
% Main loop
%% ===============================
while true
    pause(0.05);

    simStatus = get_param(model, 'SimulationStatus');
    if strcmp(simStatus, 'stopped')
        disp('[GAME2] Simulation finished.');
        % --- Retained result export -------------------------------------
        % Auto-save the full base workspace (the plant trajectory) so that
        % perf_plant_bsm2 can be run later even if MATLAB is closed or
        % crashes. A lost in-memory trajectory previously forced a full
        % 609-day re-simulation; this guarantees it never happens again.
        try
            traj_dir = fullfile(RL_DIR, 'logs');
            if ~isfolder(traj_dir), mkdir(traj_dir); end
            traj_file = fullfile(traj_dir, ...
                ['game2_traj_' datestr(now, 'yyyymmdd_HHMMSS') '.mat']);
            save(traj_file, '-v7.3');
            fprintf('[GAME2] Trajectory auto-saved to %s\n', traj_file);
        catch saveErr
            warning('[GAME2] Trajectory auto-save FAILED: %s', saveErr.message);
        end
        break
    end

    if strcmp(simStatus, 'running') || strcmp(simStatus, 'paused')
        t_sim = get_param(model, 'SimulationTime');

        if t_sim + tol >= next_pause_time
            if strcmp(get_param(model, 'SimulationStatus'), 'running')
                set_param(model, 'SimulationCommand', 'pause');
            end

            t_p = tic;
            while ~strcmp(get_param(model, 'SimulationStatus'), 'paused')
                pause(0.02);
                if toc(t_p) > 10
                    warning('[GAME2] Timeout waiting for pause.');
                    break
                end
            end

            t_sim = get_param(model, 'SimulationTime');
            fprintf('\n[GAME2] t = %.4f days\n', t_sim);

            %% --- Collect shared observations ---
            t_vars = tic;
            while ~(exist('reac1', 'var') == 1 && exist('reac2', 'var') == 1 && ...
                    exist('reac3', 'var') == 1 && exist('reac4', 'var') == 1 && ...
                    exist('reac5', 'var') == 1 && exist('settler', 'var') == 1 && ...
                    exist('in', 'var') == 1 && exist('VOL1', 'var') == 1 && ...
                    exist('VOL2', 'var') == 1 && exist('VOL3', 'var') == 1 && ...
                    exist('VOL4', 'var') == 1 && exist('VOL5', 'var') == 1 && ...
                    ~isempty(reac1) && ~isempty(reac2) && ~isempty(reac3) && ...
                    ~isempty(reac4) && ~isempty(reac5) && ~isempty(settler))
                pause(0.05);
                if toc(t_vars) > 10
                    error('[GAME2] Required BSM2 workspace variables did not become available.');
                end
            end

            SNO_1 = reac1(end, 9);
            SNO_2 = reac2(end, 9);
            SNO_3 = reac3(end, 9);
            SNH_2 = reac2(end, 10);
            SNH_4 = reac4(end, 10);
            SNH_5 = reac5(end, 10);
            SO_4  = reac4(end, 8);
            SO_5  = reac5(end, 8);
            TSS_3 = reac3(end, 14);
            TSS_5 = reac5(end, 14);

            SS_in = in(3);
            SI_in = in(2);
            SNH_in = in(11);
            CODTN_raw = (SS_in + SI_in) / max(SNH_in, 1.0);
            CODTN = min(max(CODTN_raw, 0), 100);

            Flow = game2_value_or_default('Qin', 20648.0);
            Temp = game2_value_or_default('T2', NaN);
            Qec = game2_value_or_default('carb1', 2.0);
            Qint = game2_value_or_default('Qintr', 61944.0);
            SO4ref = game2_value_or_default('SO4ref', 2.0);
            Qw = game2_value_or_default('Qw', game2_default_qw(t_sim));
            KLa1 = game2_value_or_default('kla1in', NaN);
            KLa2 = game2_value_or_default('kla2in', NaN);
            KLa3 = game2_value_or_default('kla3in', NaN);
            KLa4 = game2_value_or_default('kla4in', NaN);
            KLa5 = game2_value_or_default('kla5in', NaN);
            Qr = game2_array_value_or_default('settler', 15, 20648.0);

            Eff_SI = game2_array_value_or_default('effluent', 1, NaN);
            Eff_SS = game2_array_value_or_default('effluent', 2, NaN);
            Eff_XI = game2_array_value_or_default('effluent', 3, NaN);
            Eff_XS = game2_array_value_or_default('effluent', 4, NaN);
            Eff_XBH = game2_array_value_or_default('effluent', 5, NaN);
            Eff_XBA = game2_array_value_or_default('effluent', 6, NaN);
            Eff_XP = game2_array_value_or_default('effluent', 7, NaN);
            Eff_SNO = game2_array_value_or_default('effluent', 9, NaN);
            Eff_SNH = game2_array_value_or_default('effluent', 10, NaN);
            Eff_SND = game2_array_value_or_default('effluent', 11, NaN);
            Eff_XND = game2_array_value_or_default('effluent', 12, NaN);
            Eff_TSS = game2_array_value_or_default('effluent', 14, NaN);
            Eff_Q = game2_array_value_or_default('effluent', 15, NaN);

            n_day = min([size(reac3,1), size(reac5,1), size(settler,1), FAST_SAMPLES_PER_DAY]);
            idx_day_r = (size(reac5,1) - n_day + 1):size(reac5,1);
            idx_day_s = (size(settler,1) - n_day + 1):size(settler,1);

            n_srt = min([size(reac1,1), size(reac2,1), size(reac3,1), ...
                         size(reac4,1), size(reac5,1), size(settler,1), ...
                         SRT_WINDOW_SAMPLES]);
            idx_srt_r = (size(reac5,1) - n_srt + 1):size(reac5,1);
            idx_srt_s = (size(settler,1) - n_srt + 1):size(settler,1);

            TSS_5_1d = mean(reac5(idx_day_r, 14));
            SND_5_1d = mean(reac5(idx_day_r, 11));
            SNH_5_1d = mean(reac5(idx_day_r, 10));
            SNO_3_1d = mean(reac3((size(reac3,1) - n_day + 1):size(reac3,1), 9));
            EQI_1d = 20648.0 * (30.0 * SNH_5_1d + 10.0 * SNO_3_1d) / 1000.0;
            Qw_mean_1d = mean(settler(idx_day_s, 22));

            TSSvecreactor = ...
                reac1(idx_srt_r, 14) * VOL1 + ...
                reac2(idx_srt_r, 14) * VOL2 + ...
                reac3(idx_srt_r, 14) * VOL3 + ...
                reac4(idx_srt_r, 14) * VOL4 + ...
                reac5(idx_srt_r, 14) * VOL5;
            Qwasteflow = settler(idx_srt_s, 22);
            TSSwasteconc = settler(idx_srt_s, 53);
            TSSevec2 = settler(idx_srt_s, 36) .* settler(idx_srt_s, 37);
            removed_tss = max(TSSwasteconc .* Qwasteflow + TSSevec2, 1e-6);
            SRT_3d = mean(TSSvecreactor ./ removed_tss);

            fprintf('[GAME2] SNO2=%.3f SNH5=%.3f SO4=%.3f TSS5=%.1f CODTN=%.2f\n', ...
                SNO_2, SNH_5, SO_4, TSS_5, CODTN);
            fprintf('[GAME2] Applied actions: Qec=%.3f Qint=%.1f SO4ref=%.3f Qw=%.3f\n', ...
                Qec, Qint, SO4ref, Qw);

            T_csv = table( ...
                t_sim, SNO_1, SNO_2, SNO_3, SNH_2, SNH_4, SNH_5, ...
                SO_4, SO_5, TSS_3, TSS_5, Flow, CODTN, Temp, ...
                Qec, Qint, SO4ref, Qw, ...
                KLa1, KLa2, KLa3, KLa4, KLa5, Qr, ...
                Eff_Q, Eff_SI, Eff_SS, Eff_XI, Eff_XS, Eff_XBH, Eff_XBA, Eff_XP, ...
                Eff_SNO, Eff_SNH, Eff_SND, Eff_XND, Eff_TSS, ...
                TSS_5_1d, SND_5_1d, SRT_3d, SNH_5_1d, EQI_1d, Qw_mean_1d, SNO_3_1d, ...
                'VariableNames', { ...
                    'time','SNO_1','SNO_2','SNO_3','SNH_2','SNH_4','SNH_5', ...
                    'SO_4','SO_5','TSS_3','TSS_5','Flow','CODTN','Temp', ...
                    'Qec','Qint','SO4ref','Qw', ...
                    'KLa1','KLa2','KLa3','KLa4','KLa5','Qr', ...
                    'Eff_Q','Eff_SI','Eff_SS','Eff_XI','Eff_XS','Eff_XBH','Eff_XBA','Eff_XP', ...
                    'Eff_SNO','Eff_SNH','Eff_SND','Eff_XND','Eff_TSS', ...
                    'TSS_5_1d','SND_5_1d','SRT_3d','SNH_5_1d','EQI_1d','Qw_mean_1d','SNO_3_1d'});
            writetable(T_csv, STATE_FILE);

            if isfile(FLAG_STATE), delete(FLAG_STATE); end
            fid = fopen(FLAG_STATE, 'w'); fclose(fid);
            disp('[GAME2] flag_state created');

            %% --- Wait for Python action row ---
            disp('[GAME2] Waiting for Python action row...');
            t_wait = tic;
            while ~isfile(FLAG_ACTION)
                if isfile(FLAG_ERROR)
                    if isfile(FLAG_STATE), delete(FLAG_STATE); end
                    error('[GAME2] Python coordinator reported fatal error at t=%.4f days.', t_sim);
                end
                pause(0.05);
                if toc(t_wait) > TIMEOUT
                    if ABORT_ON_TIMEOUT
                        if isfile(FLAG_STATE), delete(FLAG_STATE); end
                        error('[GAME2] TIMEOUT waiting for Python action at t=%.4f days.', t_sim);
                    end
                    warning('[GAME2] TIMEOUT - writing default BSM2 action row.');
                    Tdef = table(2.0, 61944.0, 2.0, game2_default_qw(t_sim), ...
                        'VariableNames', {'Qec','Qint','SO4ref','Qw'});
                    writetable(Tdef, ACTION_FILE);
                    fid = fopen(FLAG_ACTION, 'w'); fclose(fid);
                    break
                end
            end
            disp('[GAME2] Action row received');

            update_game2_actions_from_python(ACTION_FILE);

            delete(FLAG_ACTION);
            disp('[GAME2] flag_action consumed');

            next_pause_time = next_pause_time + dt;
            set_param(model, 'SimulationCommand', 'continue');
            disp('[GAME2] Simulation resumed');
        end
    end
end

% Clean shutdown: discard the dirty (accelerator-built) model so the headless
% -batch session does not warn about unsaved changes when it exits.
if bdIsLoaded(model)
    try
        close_system(model, 0);
    catch
    end
end


function value = game2_value_or_default(name, default_value)
    value = default_value;
    if evalin('base', sprintf('exist(''%s'', ''var'')', name))
        candidate = evalin('base', name);
        if isnumeric(candidate) && ~isempty(candidate) && isfinite(candidate(1))
            value = candidate(end);
        end
    end
end


function value = game2_array_value_or_default(name, col, default_value)
    value = default_value;
    if evalin('base', sprintf('exist(''%s'', ''var'')', name))
        candidate = evalin('base', name);
        if isnumeric(candidate) && ~isempty(candidate) && size(candidate, 2) >= col
            raw = candidate(end, col);
            if isfinite(raw)
                value = raw;
            end
        end
    end
end


function qw = game2_default_qw(t_sim)
    if t_sim < 182
        qw = 300.0;
    elseif t_sim < 364
        qw = 450.0;
    elseif t_sim < 546
        qw = 300.0;
    else
        qw = 450.0;
    end
end
