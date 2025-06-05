using System;
using System.Collections.Generic;

public class Neuron
{
    static Random random = new Random();

    public static void Main(string[] args)
    {
        int ZDmin = -5, ZDmax = 5;
        double parametrUczenia = 0.3; 

        int iloscWag = 9;

        int beta = 1;

        double[] wagi = new double[iloscWag];
        for (int i = 0; i < iloscWag; i++)
        {
            wagi[i] = random.NextDouble() * (ZDmax - ZDmin) + ZDmin;
        }

        Console.WriteLine("Wylosowane wagi:");
        for (int i = 0; i < iloscWag; i++)
        {
            if (i == 0 || i == 1 || i == 3 || i == 4 || i == 6 || i == 7)
            {
                Console.WriteLine($"Waga {i} = {wagi[i]:F4}");
            }
            else if (i == 2 || i == 5 || i == 8)
            {
                Console.WriteLine($"Bias {i} = {wagi[i]:F4}");
            }
        }

        List<(int[] wejscie, int oczekiwane)> dane = new List<(int[], int)>
        {
            (new int[] {0, 0}, 0),
            (new int[] {0, 1}, 1),
            (new int[] {1, 0}, 1),
            (new int[] {1, 1}, 0)
        };
        int liczbaEpok = 50000;

        Console.WriteLine("\nModuł błędu dla każdej próbki (samokontrola):");
        foreach (var (wejscie, oczekiwane) in dane)
        {
            double suma1 = wagi[0] * wejscie[0] + wagi[1] * wejscie[1] + wagi[2];
            double suma2 = wagi[3] * wejscie[0] + wagi[4] * wejscie[1] + wagi[5];

            double wyjscie1 = Sigmoid(suma1, beta);
            double wyjscie2 = Sigmoid(suma2, beta);

            double suma_wyjscie = wagi[6] * wyjscie1 + wagi[7] * wyjscie2 + wagi[8];
            double finalne_wyjscie = Sigmoid(suma_wyjscie, beta);

            double modul_bledu = Math.Abs(oczekiwane - finalne_wyjscie);

            Console.WriteLine($"Wejście: [{wejscie[0]}, {wejscie[1]}], Wyjście: {finalne_wyjscie:F4}, Oczekiwane: {oczekiwane}, Moduł błędu: {modul_bledu:F4}");
        }


        for (int epoch = 0; epoch < liczbaEpok; epoch++)
        {

            for (int i = dane.Count - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                var temp = dane[i];
                dane[i] = dane[j];
                dane[j] = temp;
            }
            foreach (var (wejscie, oczekiwane) in dane)
            {
                double suma1 = wagi[0] * wejscie[0] + wagi[1] * wejscie[1] + wagi[2];
                double suma2 = wagi[3] * wejscie[0] + wagi[4] * wejscie[1] + wagi[5];

                double wyjscie1 = Sigmoid(suma1, beta);
                double wyjscie2 = Sigmoid(suma2, beta);

                double suma_wyjscie = wagi[6] * wyjscie1 + wagi[7] * wyjscie2 + wagi[8];
                double finalne_wyjscie = Sigmoid(suma_wyjscie, beta);

                double blad = oczekiwane - finalne_wyjscie;

                double pochodna_wyjscie = blad * parametrUczenia*finalne_wyjscie * (1 - finalne_wyjscie);

                double korekta_w6 = pochodna_wyjscie * wyjscie1;
                double korekta_w7 = pochodna_wyjscie * wyjscie2;
                double korekta_b8 = pochodna_wyjscie;

                double pochodna1 = wyjscie1 * (1 - wyjscie1);
                double pochodna2 = wyjscie2 * (1 - wyjscie2);

                double delta1 = pochodna1 * wagi[6] * pochodna_wyjscie;
                double delta2 = pochodna2 * wagi[7] * pochodna_wyjscie;

                double korekta_w0 =  delta1 * wejscie[0];
                double korekta_w1 =  delta1 * wejscie[1];
                double korekta_b2 =  delta1;

                double korekta_w3 =  delta2 * wejscie[0];
                double korekta_w4 =  delta2 * wejscie[1];
                double korekta_b5 =  delta2;

                wagi[0] += korekta_w0;
                wagi[1] += korekta_w1;
                wagi[2] += korekta_b2;
                wagi[3] += korekta_w3;
                wagi[4] += korekta_w4;
                wagi[5] += korekta_b5;
                wagi[6] += korekta_w6;
                wagi[7] += korekta_w7;
                wagi[8] += korekta_b8;
            }
            if (epoch % 10000 == 0)
            {
                Console.WriteLine($"\n--- Epoka {epoch} ---");
                foreach (var (wejscie, oczekiwane) in dane)
                {
                    double suma1 = wagi[0] * wejscie[0] + wagi[1] * wejscie[1] + wagi[2];
                    double suma2 = wagi[3] * wejscie[0] + wagi[4] * wejscie[1] + wagi[5];

                    double wyjscie1 = Sigmoid(suma1, beta);
                    double wyjscie2 = Sigmoid(suma2, beta);

                    double suma_wyjscie = wagi[6] * wyjscie1 + wagi[7] * wyjscie2 + wagi[8];
                    double finalne_wyjscie = Sigmoid(suma_wyjscie, beta);

                    double modul_bledu = Math.Abs(oczekiwane - finalne_wyjscie);

                    Console.WriteLine($"Wejście: [{wejscie[0]}, {wejscie[1]}], Wyjście: {finalne_wyjscie:F4}, Oczekiwane: {oczekiwane}, Błąd: {modul_bledu:F4}");
                }
            }
        }

        Console.WriteLine("\nWyniki po treningu:");
        foreach (var (wejscie, oczekiwane) in dane)
        {
            double suma1 = wagi[0] * wejscie[0] + wagi[1] * wejscie[1] + wagi[2];
            double suma2 = wagi[3] * wejscie[0] + wagi[4] * wejscie[1] + wagi[5];

            double wyjscie1 = Sigmoid(suma1, beta);
            double wyjscie2 = Sigmoid(suma2, beta);

            double suma_wyjscie = wagi[6] * wyjscie1 + wagi[7] * wyjscie2 + wagi[8];
            double finalne_wyjscie = Sigmoid(suma_wyjscie, beta);

            Console.WriteLine($"Wejście: [{wejscie[0]}, {wejscie[1]}] => Wyjście: {finalne_wyjscie:F4}, Oczekiwane: {oczekiwane}");
        }

        Console.WriteLine("\nOstateczne wagi:");
        for (int i = 0; i < iloscWag; i++)
        {
            if (i == 0 || i == 1 || i == 3 || i == 4 || i == 6 || i == 7)
                Console.WriteLine($"Waga {i} = {wagi[i]:F4}");
            else if (i == 2 || i == 5 || i == 8)
                Console.WriteLine($"Bias {i} = {wagi[i]:F4}");
        }

       
    }

    public static double Sigmoid(double x, double beta)
    {
        return 1 / (1 + Math.Exp(-beta * x));
    }
}