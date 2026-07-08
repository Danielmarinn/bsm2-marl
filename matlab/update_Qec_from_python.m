function update_Qec_from_python(action_file)
% UPDATE_QEC_FROM_PYTHON
%   Le o caudal de carbono externo (Qec) do action.csv escrito pelo
%   Python e aplica-o a variavel carb1 do workspace MATLAB.
%
%   carb1 e' a variavel standard do BSM2 para o doseamento de
%   carbono externo no reactor 1 (anoxico) — ponto de doseamento
%   para melhoria da desnitrificacao. Confirmada via `whos carb*`
%   apos init_bsm2: scalar 1x1 double.
%
%   Limites BSM2 (Table 18, "Available control handles"):
%       Qec in [0, 5] m3/d
%
%   Concentracao do carbono externo: CARBONSOURCECONC = 400000 g COD/m3
%   (constante no workspace — nao alterada por este agente).
%
%   Default BSM2 (Section 9, "Initialization"): Qec = 2 m3/d.

    QEC_MIN = 0.0;
    QEC_MAX = 5.0;

    try
        data = readmatrix(action_file);
        data = data(~isnan(data));
        Qec = data(end);
        Qec = max(QEC_MIN, min(QEC_MAX, Qec));
    catch e
        warning('update_Qec_from_python: %s — usando default BSM2 (2 m3/d)', e.message);
        Qec = 2.0;
    end

    % Aplica ao reactor 1 (ponto de doseamento anoxico)
    assignin('base', 'carb1', Qec);

    fprintf('[RL] Qec = %.3f m3/d (carb1)\n', Qec);

end
