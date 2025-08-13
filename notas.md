## Dias de interes
- 18 enero 2024 nubosidad
- 16 enero 2024 lluvia
- 7 febrero 2024 dia normal soleado de verano
- 13 agosto 2024 dia normal de invierno

## Notas de interes
- Se agregó un dummy de mes como feature (analizar modelo con y sin)
- MPPT energy (Para una variable acumulativa como la energía total, que funciona como el odómetro de un auto, no te interesa el valor total en cada momento, sino cuánto ha aumentado entre una medición y la siguiente.) por eso hice la diff
- Primeros entrenamientos muestran una suavización, problema clasico en prediccion de series temporales (MSE)