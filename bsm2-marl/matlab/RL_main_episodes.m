%% matlab/RL_main_episodes.m — Orquestrador com episódios curtos
%
% ANTES DE CORRER:
%   1. init_bsm2
%   2. load('states_day245.mat')
%   3. Terminal: python agents/ctrl_sac_qint.py
%   4. run este script

model    = 'bsm2_cl';
BSM2_DIR = 'C:/Users/marin/Documents/BSM2_R2019b';
COMMS    = 'C:/Users/marin/Documents/BSM2_R2019b/prog_RL/comms';
addpath('C:/Users/marin/Documents/BSM2_R2019b/prog_RL/matlab');

%% ===============================
% Parâmetros de episódio
%% ===============================
EPISODE_DAYS = 50;
START_DAY    = 245;
STOP_DAY     = START_DAY + EPISODE_DAYS;   % 295
N_EPISODES   = 20;

%% ===============================
% Parâmetros de tempo
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
% Verificações iniciais
%% ===============================
if ~isfile(STATE_DAY245)
    error('[RL] states_day245.mat nao encontrado!');
end

if ~bdIsLoaded(model)
    open_system(model);
    fprintf('[RL] Modelo %s carregado.\n', model);
end

%% ===============================
% Limpar flags antigos
%% ===============================
for f = {FLAG_STATE, FLAG_ACTION, FLAG_EPISODE}
    if isfile(f{1}), delete(f{1}); end
end

fprintf('\n[RL] Iniciando treino: %d episodios x %d dias\n\n', N_EPISODES, EPISODE_DAYS);

%% ===============================
% LOOP DE EPISÓDIOS
%% ===============================
for ep = 1:N_EPISODES

    fprintf('\n%s\n[RL] EPISODIO %d / %d\n%s\n', ...
            repmat('=',1,50), ep, N_EPISODES, repmat('=',1,50));

    %% --- Reset: carregar estado do dia 245 ---
    load(STATE_DAY245);
    fprintf('[RL] Estado do dia 245 carregado.\n');

    %% --- Configurar simulação ---
    set_param(model, 'SimulationMode', 'accelerator');
    set_param(model, 'StartTime', num2str(START_DAY));
    set_param(model, 'StopTime',  num2str(STOP_DAY));
    set_param(model, 'OutputTimes', ...
        ['[' num2str(START_DAY) ':(1/96):' num2str(STOP_DAY) ']']);

    %% --- Sinalizar Python: novo episódio ---
    T_ep = table(ep, START_DAY, STOP_DAY, N_EPISODES, ...
        'VariableNames', {'episode','start_day','stop_day','total_episodes'});
    writetable(T_ep, EPISODE_FILE);
    fid = fopen(FLAG_EPISODE, 'w'); fclose(fid);
    fprintf('[RL] flag_episode criado (ep=%d)\n', ep);

    %% --- Iniciar simulação ---
    set_param(model, 'SimulationCommand', 'start');
    fprintf('[RL] Simulacao iniciada (t=%d -> t=%d)\n', START_DAY, STOP_DAY);

    pause(1);
    t_init = tic;
    while strcmp(get_param(model, 'SimulationStatus'), 'initializing')
        pause(0.1);
        if toc(t_init) > 120
            error('[RL] Timeout na inicializacao do modelo.');
        end
    end
    fprintf('[RL] Status: %s\n', get_param(model, 'SimulationStatus'));

    % IMPORTANTE: next_pause_time = START_DAY + dt
    next_pause_time = START_DAY + dt;

    %% --- Loop interno do episódio ---
    while true

        pause(0.05);

        simStatus = get_param(model, 'SimulationStatus');

        if strcmp(simStatus, 'stopped')
            fprintf('[RL] Episodio %d concluido.\n', ep);
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
                        warning('[RL] Timeout a aguardar pausa.');
                        break
                    end
                end

                t_sim = get_param(model, 'SimulationTime');

                %% --- Coletar observações CTRL-2 ---
                % reac1/2/3 são arrays actualizados pela simulação
                % S_NO1 etc. são valores de inicialização — NÃO usar
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

                %% --- Escrever state.csv ---
                T_csv = table(SNO_2, SNO_1, SNO_3, CODTN, SNH_2, Flow, Temp, t_sim, ...
                    'VariableNames', ...
                    {'SNO_2','SNO_1','SNO_3','CODTN','SNH_in','Flow','Temp','time'});
                writetable(T_csv, STATE_FILE);

                if isfile(FLAG_STATE), delete(FLAG_STATE); end
                fid = fopen(FLAG_STATE, 'w'); fclose(fid);

                %% --- Aguardar ação do Python ---
                t_wait = tic;
                while ~isfile(FLAG_ACTION)
                    pause(0.05);
                    if toc(t_wait) > TIMEOUT
                        warning('[RL] TIMEOUT — Qint default aplicado.');
                        Tdef = table(61944.0, 'VariableNames', {'Qec'});
                        writetable(Tdef, ACTION_FILE);
                        fid = fopen(FLAG_ACTION, 'w'); fclose(fid);
                        break
                    end
                end

                %% --- Aplicar ação ---
                update_Qint_from_python(ACTION_FILE);
                delete(FLAG_ACTION);

                %% --- Retomar ---
                next_pause_time = next_pause_time + dt;
                set_param(model, 'SimulationCommand', 'continue');

            end
        end
    end

    if ep < N_EPISODES
        pause(2);
    end

end

fprintf('\n[RL] Treino completo! %d episodios concluidos.\n', N_EPISODES);