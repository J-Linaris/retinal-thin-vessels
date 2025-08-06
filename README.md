# retinal_thin_vessels

AUTOR:

João Paulo Menezes Linaris
joaolinaris@gmail.com

FUNCTIONS:

    get_thin_vessels: works with any segmentation mask. Also, it supports mask_type argument,
                      so you can pass the probabilities score mask produced by Sigmoid directly
                      to the function and it will show the filtered mask.

COMO EXECUTAR:

Para executar o programa "uspsh.c", primeiro, verifique que o executável
"uspsh" está presente no seu diretório atual. Se estiver, rode:

$./uspsh

Após executar este comando, algo como:

[eclipse:/tmp]$

deve aparecer no seu terminal.

Para executar o programa "uspsh.c", primeiro, verifique que o executável
"uspsh" está presente no seu diretório atual. Se estiver, rode:

$./ep1 número_algoritmo arquivo_trace arquivo_resultados

Onde:

    número_algoritmo: número de 1 a 3 indicando qual algoritmo se deseja rodar:                
                    1: First Come First Saved (FCFS)
                    2: Shortest Remaining Time Next (SRTN)
                    3: Escalonamento com prioridade
    arquivo_trace: arquivo com os processos a serem executados pelo simulador
    arquivo_resultados: arquivo em que será gravado os números obtidos para 
                        cada processo após a simulação mais uma linha extra
                        com o número de preempções realizadas pelo algoritmo
                        escalonador.

Exemplo de execução:

$./ep1 1 entrada-esperado.txt resultados.txt

DEPENDÊNCIAS:

    Para rodar o programa, recomenda-se o uso do Sistema Operacional Ubuntu 22.04.2 LTS,
uma vez que foi desenvolvido e testado nesse Sistema Operacional. Além disso, é
necessário ter instalado o compilador de programas em linguagem C: "gcc", cuja versão
usada para desenvolver e testar o programa foi a 11.4.0 .