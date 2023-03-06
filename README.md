
## Como rodar o programa
O script `calculate_disparities.py` realiza o cálculo das disparidades entre duas imagens. É considerado que as duas imagens são capturadas por duas câmeras e de uma mesma cena, e que cada pixel em uma linha da primeira imagem projeta para um pixel na mesma linha da segunda imagem. As disparidades calculadas são salvas no subdiretório `/disparities`, localizado no mesmo diretório que o script e com o último cálculo realizado com um conjunto específico de parâmetros sendo armazenado em `/disparities/latest`.

O script `compare_gt.py` busca por uma imagem fornecida via linha de comando em `/disparities/latest` e realiza a análise quantitativa dos mapas de disparidade.

Ambos scripts recebem argumentos na linha de comando, os quais podem ser consultados com o argumento `-h`