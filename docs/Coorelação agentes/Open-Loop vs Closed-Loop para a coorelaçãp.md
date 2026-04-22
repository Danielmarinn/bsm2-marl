#### &#x09;**Open-Loop vs Closed-Loop para correlação dos agentes**





Open-Loop: O sistema corre sem controlador ativo — as variáveis manipuladas (Qec, Qint, DOref, Qw) ficam fixas ou seguem um perfil predefinido. Isto significa que o processo responde livremente às perturbações do afluente, e as correlações que observas refletem as relações naturais do processo sem interferência de nenhum controlador.



Closed-Loop: O controlador está ativo e compensa continuamente as perturbações. Isto significa que por exemplo o SNO se mantém relativamente estável porque o controlador já está a ajustar o Qec. O resultado é que as correlações ficam artificialmente atenuadas — o controlador está a destruir exatamente a variabilidade que querias observar.



MESMO ASSIM ESCOLHEMOS CLOSED-LOOP:

“Although the BSM2 benchmark includes a dissolved oxygen controller, it does not actively regulate key manipulated variables such as Qec, Qint, or Qw. Therefore, the system retains sufficient open-loop characteristics for the purpose of variable selection and sensitivity analysis.”

