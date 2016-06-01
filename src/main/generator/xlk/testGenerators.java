package main.generator.xlk;

import com.opencsv.CSVWriter;

import java.io.FileWriter;
import java.io.IOException;

/**
 * Created by loongkuan on 31/05/16.
 */
public class testGenerators {
    public static void main(String[] args){

    }

    public static void writeToCSV(String[][] data, String[] header, String filename) throws IOException {
        CSVWriter writer = new CSVWriter(new FileWriter(filename), ',');
        // feed in your array (or convert your data to an array)
        writer.writeNext(header);
        for (String[] dataLine : data) {
            writer.writeNext(dataLine);
        }
        writer.close();
    }

    private static String[][] classifierDistanceTest(JointModel baseModel,
                                                     dataStream) {
        double[] driftMags = new double[]{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7};
        int[] dataPoints = new int []{100, 1000, 10000, 100000};
        int nTests = 10;

        String[][] results = new String[driftMags.length][dataPoints.length];

        for (int i = 0; i < driftMags.length; i++) {
            double dm = driftMags[i];
            dataStream.driftMagnitudePrior.setValue(dm);
            dataStream.driftMagnitudeConditional.setValue(dm);

            for (int j = 0; j < dataPoints.length; j++) {
                int nData = dataPoints[j];
                dataStream.burnInNInstances.setValue(nData);
                //System.out.println("Drift Magnitude: " + dm);
                //System.out.println("No. Instances Before/After Drift: " + nData);

                double avgPygv = 0.0f;
                for (int k = 0; k < nTests; k++) {
                    //System.out.println("Run: " + (k+1) + "\t");
                    dataStream.restart();
                    dataStream.prepareForUse();
                    Experiments experiment = new Experiments(baseModel.copy(), dataStream);
                    avgPygv += Double.parseDouble(experiment.distanceBetweenStartEnd()[0]);
                }
                avgPygv /= nTests;
                //System.out.println("p(y|X) drift = " + avgPygv);
                results[i][j] = Double.toString(avgPygv);
                System.out.println("Estimated p(X): " + avgPygv);
            }
        }

        return results;
    }

}
