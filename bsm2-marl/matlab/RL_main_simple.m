%% matlab/RL_main_simple.m — Orquestrador simples (t=0 a t=609)
%
% IMPORTANTE: corre desde t=0 (não desde t=245).
% Correr desde t=245 em accelerator completa instantaneamente
% sem dar tempo ao loop para pausar.
% Os primeiros 245 dias são estabilização — o SAC aprende
% apenas nos dias 245-609 mas a simulação tem de correr desde t=0.
%
% ANTES DE CORRER:
%   1. init_bsm2
%   2. Terminal: python agents/ctrl_sac_qint.py
%   3. run este script
%   (NÃO carregar states_day245.mat)

%% matlab/RL_main_simple.m
cd('C:/Users/marin/Documents/BSM2_R2019b');
addpath('C:/Users/marin/Documents/BSM2_R2019b');

model    = 'bsm2_cl';
COMMS    = 'C:/Users/marin/Documents/BSM2_R2019b/prog_RL/comms';
addpath('C:/Users/marin/Documents/BSM2_R2019b/prog_RL/matlab');

%% ===============================
% Parâmetros de tempo
%% ===============================
dt      = 15 / (24*60);   % 15 minutos em dias
tol     = 1e-6;
TIMEOUT = 30;

%% ===============================
% Paths
%% ===============================
FLAG_STATE  = fullfile(COMMS, 'flag_state.run');
FLAG_ACTION = fullfile(COMMS, 'flag_action.run');
STATE_FILE  = fullfile(COMMS, 'state.csv');
ACTION_FILE = fullfile(COMMS, 'action.csv');

%% ===============================
% Limpar flags antigos
%% ===============================
disp('[RL_main] Limpando flags antigos...');
if isfile(FLAG_STATE),  delete(FLAG_STATE);  end
if isfile(FLAG_ACTION), delete(FLAG_ACTION); end

%% ===============================
% Configurar simulação — DESDE t=0
%% ===============================
set_param('bsm2_cl', 'SimulationMode', 'accelerator');
set_param('bsm2_cl', 'StartTime',   '0');
set_param('bsm2_cl', 'StopTime',    '609');
set_param('bsm2_cl', 'OutputTimes', '[0:(1/96):609]');
set_param('bsm2_cl', 'SimulationCommand', 'start');
disp('[RL_main] Simulacao iniciada (t=0 -> t=609)...');

% Aguardar arranque
pause(0.5);
while strcmp(get_param('bsm2_cl', 'SimulationStatus'), 'initializing')
    pause(0.1);
end

% Primeira pausa em t=dt (~0.01 dias) — funciona com accelerator
next_pause_time = dt;

%% ===============================
% Loop principal
%% ===============================
while true

    pause(0.05);

    simStatus = get_param('bsm2_cl', 'SimulationStatus');
    if strcmp(simStatus, 'stopped')
        disp('[RL_main] Simulacao finalizada.');
        break
    end

    if strcmp(simStatus, 'running') || strcmp(simStatus, 'paused')

        t_sim = get_param('bsm2_cl', 'SimulationTime');

        if t_sim + tol >= next_pause_time

            if strcmp(get_param('bsm2_cl', 'SimulationStatus'), 'running')
                set_param('bsm2_cl', 'SimulationCommand', 'pause');
            end

            t_p = tic;
            while ~strcmp(get_param('bsm2_cl', 'SimulationStatus'), 'paused')
                pause(0.02);
                if toc(t_p) > 10
                    warning('[RL_main] Timeout a aguardar pausa.');
                    break
                end
            end

            t_sim = get_param('bsm2_cl', 'SimulationTime');
            fprintf('\n[RL_main] t = %.4f dias\n', t_sim);

            %% --- Coletar observações CTRL-2 ---
            % reac1/2/3: arrays actualizados pelo Simulink (To Workspace)
            % NÃO usar S_NO1/S_NO2 — são escalares de inicialização
            SNO_1 = reac1(end, 9);    % nitrato reactor 1 (anóxico)
            SNO_2 = reac2(end, 9);    % nitrato reactor 2 (anóxico)
            SNO_3 = reac3(end, 9);    % nitrato reactor 3 (aeróbio)
            SNH_2 = reac2(end, 10);   % amónia reactor 2

            % in: vector 1x21 do influente actual (actualizado)
            SS_in  = in(3);
            SI_in  = in(2);
            SNH_in = in(11);
            CODTN  = (SS_in + SI_in) / (SNH_in + 1e-6);

            Flow = Qin;
            Temp = T2;

            fprintf('[RL_main] SNO2=%.4f SNO1=%.4f SNO3=%.4f CODTN=%.2f SNH=%.4f\n', ...
                SNO_2, SNO_1, SNO_3, CODTN, SNH_2);

            %% --- Escrever state.csv ---
            T_csv = table(SNO_2, SNO_1, SNO_3, CODTN, SNH_2, Flow, Temp, t_sim, ...
                'VariableNames', ...
                {'SNO_2','SNO_1','SNO_3','CODTN','SNH_in','Flow','Temp','time'});
            writetable(T_csv, STATE_FILE);

            if isfile(FLAG_STATE), delete(FLAG_STATE); end
            fid = fopen(FLAG_STATE, 'w'); fclose(fid);
            disp('[RL_main] flag_state criado');

            %% --- Aguardar ação do Python ---
            disp('[RL_main] Aguardando acao do Python...');
            t_wait = tic;
            while ~isfile(FLAG_ACTION)
                pause(0.05);
                if toc(t_wait) > TIMEOUT
                    warning('[RL_main] TIMEOUT — Qint default.');
                    Tdef = table(61944.0, 'VariableNames', {'Qec'});
                    writetable(Tdef, ACTION_FILE);
                    fid = fopen(FLAG_ACTION, 'w'); fclose(fid);
                    break
                end
            end
            disp('[RL_main] Acao recebida');

            %% --- Aplicar ação ---
            update_Qint_from_python(ACTION_FILE);
            delete(FLAG_ACTION);
            disp('[RL_main] flag_action consumido');

            %% --- Retomar ---
            next_pause_time = next_pause_time + dt;
            set_param('bsm2_cl', 'SimulationCommand', 'continue');
            disp('[RL_main] Simulacao retomada');

        end
    end
end