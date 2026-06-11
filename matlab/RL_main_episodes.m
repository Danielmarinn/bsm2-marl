%% matlab/RL_main_episodes.m — LEGACY: Qint-only episodic orchestrator
%  Superseded by RL_main_simple.m (single-agent) and RL_main_game2.m (multi-agent).
%  Kept for reference only.
%
% BEFORE RUNNING:
%   1. init_bsm2
%   2. load('states_day245.mat')
%   3. Terminal: python agents/ctrl_sac_qint.py
%   4. run this script

model    = 'bsm2_cl';
RL_DIR   = fileparts(fileparts(mfilename('fullpath')));
THESIS_DIR = fileparts(RL_DIR);
UNI_DIR    = fileparts(THESIS_DIR);
BSM2_DIR   = fullfile(THESIS_DIR, 'BSM2_R2019b');
if ~isfolder(BSM2_DIR)
    BSM2_DIR = fullfile(UNI_DIR, 'BSM2_R2019b');
end
if ~isfolder(BSM2_DIR)
    error('[RL_main_episodes] BSM2_R2019b folder not found. Checked Thesis work and Universidade folders.');
end
COMMS    = fullfile(RL_DIR, 'comms');
addpath(fullfile(RL_DIR, 'matlab'));

%% ===============================
% Episode parameters
%% ===============================
EPISODE_DAYS = 50;
START_DAY    = 245;
STOP_DAY     = START_DAY + EPISODE_DAYS;   % 295
N_EPISODES   = 20;

%% ===============================
% Time parameters
%% ===============================
dt      = 15 / (24*60);
tol     = 1e-6;
TIMEOUT = 30;

%% ===============================
% Paths
%% ===============================
FLAG_STATE   = fullfile(COMMS, 'flag_state.run');
FLAG_ACTION  = fullfile(COMMS, 'flag_action.run');
FLAG_EPISODE = fullfile(COMMS, 'flag_episode.run');
STATE_FILE   = fullfile(COMMS, 'state.csv');
ACTION_FILE  = fullfile(COMMS, 'action.csv');
EPISODE_FILE = fullfile(COMMS, 'episode_info.csv');
STATE_DAY245 = fullfile(BSM2_DIR, 'states_day245.mat');

%% ===============================
% Initial checks
%% ===============================
if ~isfile(STATE_DAY245)
    error('[RL] states_day245.mat not found!');
end

if ~bdIsLoaded(model)
    open_system(model);
    fprintf('[RL] Model %s loaded.\n', model);
end

%% ===============================
% Clear old flags
%% ===============================
for f = {FLAG_STATE, FLAG_ACTION, FLAG_EPISODE}
    if isfile(f{1}), delete(f{1}); end
end

fprintf('\n[RL] Starting training: %d episodes x %d days\n\n', N_EPISODES, EPISODE_DAYS);

%% ===============================
% EPISODE LOOP
%% ===============================
for ep = 1:N_EPISODES

    fprintf('\n%s\n[RL] EPISODE %d / %d\n%s\n', ...
            repmat('=',1,50), ep, N_EPISODES, repmat('=',1,50));

    %% --- Reset: load day-245 state ---
    load(STATE_DAY245);
    fprintf('[RL] Day-245 state loaded.\n');

    %% --- Configure simulation ---
    set_param(model, 'SimulationMode', 'accelerator');
    set_param(model, 'StartTime', num2str(START_DAY));
    set_param(model, 'StopTime',  num2str(STOP_DAY));
    set_param(model, 'OutputTimes', ...
        ['[' num2str(START_DAY) ':(1/96):' num2str(STOP_DAY) ']']);

    %% --- Signal Python: new episode ---
    T_ep = table(ep, START_DAY, STOP_DAY, N_EPISODES, ...
        'VariableNames', {'episode','start_day','stop_day','total_episodes'});
    writetable(T_ep, EPISODE_FILE);
    fid = fopen(FLAG_EPISODE, 'w'); fclose(fid);
    fprintf('[RL] flag_episode created (ep=%d)\n', ep);

    %% --- Start simulation ---
    set_param(model, 'SimulationCommand', 'start');
    fprintf('[RL] Simulation started (t=%d -> t=%d)\n', START_DAY, STOP_DAY);

    pause(1);
    t_init = tic;
    while strcmp(get_param(model, 'SimulationStatus'), 'initializing')
        pause(0.1);
        if toc(t_init) > 120
            error('[RL] Timeout during model initialization.');
        end
    end
    fprintf('[RL] Status: %s\n', get_param(model, 'SimulationStatus'));

    % IMPORTANT: next_pause_time = START_DAY + dt
    next_pause_time = START_DAY + dt;

    %% --- Inner episode loop ---
    while true

        pause(0.05);

        simStatus = get_param(model, 'SimulationStatus');

        if strcmp(simStatus, 'stopped')
            fprintf('[RL] Episode %d complete.\n', ep);
            break
        end

        if strcmp(simStatus, 'running') || strcmp(simStatus, 'paused')

            t_sim = get_param(model, 'SimulationTime');

            if t_sim + tol >= next_pause_time

                if strcmp(get_param(model, 'SimulationStatus'), 'running')
                    set_param(model, 'SimulationCommand', 'pause');
                end

                t_pause = tic;
                while ~strcmp(get_param(model, 'SimulationStatus'), 'paused')
                    pause(0.02);
                    if toc(t_pause) > 10
                        warning('[RL] Timeout waiting for pause.');
                        break
                    end
                end

                t_sim = get_param(model, 'SimulationTime');

                %% --- Collect CTRL-2 observations ---
                % reac1/2/3 are arrays updated by the simulation
                % S_NO1 etc. are initialization values — do NOT use
                SNO_1 = reac1(end, 9);
                SNO_2 = reac2(end, 9);
                SNO_3 = reac3(end, 9);
                SNH_2 = reac2(end, 10);

                SS_in  = in(3);
                SI_in  = in(2);
                SNH_in = in(11);
                CODTN  = (SS_in + SI_in) / (SNH_in + 1e-6);

                Flow = Qin;
                Temp = T2;

                fprintf('[ep%02d t=%.3f] SNO2=%.3f SNO1=%.3f SNO3=%.3f CODTN=%.2f SNH=%.3f\n', ...
                    ep, t_sim, SNO_2, SNO_1, SNO_3, CODTN, SNH_2);

                %% --- Write state.csv ---
                T_csv = table(SNO_2, SNO_1, SNO_3, CODTN, SNH_2, Flow, Temp, t_sim, ...
                    'VariableNames', ...
                    {'SNO_2','SNO_1','SNO_3','CODTN','SNH_in','Flow','Temp','time'});
                writetable(T_csv, STATE_FILE);

                if isfile(FLAG_STATE), delete(FLAG_STATE); end
                fid = fopen(FLAG_STATE, 'w'); fclose(fid);

                %% --- Wait for action from Python ---
                t_wait = tic;
                while ~isfile(FLAG_ACTION)
                    pause(0.05);
                    if toc(t_wait) > TIMEOUT
                        warning('[RL] TIMEOUT — Qint default applied.');
                        Tdef = table(61944.0, 61944.0, 'VariableNames', {'Qint','Qec'});
                        writetable(Tdef, ACTION_FILE);
                        fid = fopen(FLAG_ACTION, 'w'); fclose(fid);
                        break
                    end
                end

                %% --- Apply action ---
                update_Qint_from_python(ACTION_FILE);
                delete(FLAG_ACTION);

                %% --- Resume ---
                next_pause_time = next_pause_time + dt;
                set_param(model, 'SimulationCommand', 'continue');

            end
        end
    end

    if ep < N_EPISODES
        pause(2);
    end

end

fprintf('\n[RL] Training complete! %d episodes finished.\n', N_EPISODES);
