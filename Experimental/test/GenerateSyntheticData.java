import main.generator.AbruptTreeDriftGenerator;
import main.generator.CategoricalDriftGenerator;

/**
 * Created by LoongKuan on 8/10/2016.
 */
public class GenerateSyntheticData {
    public static void main(String[] args) {
        String filename;
        int nAttributes = 5;
        int nValues = 5;

        // Set data generator to use
        AbruptTreeDriftGenerator dataStream = new AbruptTreeDriftGenerator();

        dataStream.nAttributes.setValue(nAttributes);
        dataStream.nValuesPerAttribute.setValue(nValues);
        dataStream.precisionDriftMagnitude.setValue(0.01);

        int[] burnIns = new int[]{1000000};
        double[] magnitudes = new double[]{0.2, 0.5, 0.8};
        String folder = "./datasets/synthetic_" + nAttributes + "Att_" + nValues + "Val/";
        String extension = ".arff";

        for (int burnIn : burnIns) {

            dataStream.burnInNInstances.setValue(burnIn);
            dataStream.driftConditional.setValue(false);
            dataStream.driftPriors.setValue(false);
            filename = "n" + Integer.toString(burnIn) + "_none";
            dataStream.prepareForUse();
            dataStream.writeDataStreamToFile(folder + filename + extension);

            for (double magnitude : magnitudes) {
                dataStream.driftMagnitudeConditional.setValue(magnitude);
                dataStream.driftMagnitudePrior.setValue(magnitude);

                dataStream.driftConditional.setValue(false);
                dataStream.driftPriors.setValue(true);
                filename = "n" + Integer.toString(burnIn) + "_m" + Double.toString(magnitude) + "_prior";
                dataStream.prepareForUse();
                dataStream.writeDataStreamToFile(folder + filename + extension);

                dataStream.driftConditional.setValue(true);
                dataStream.driftPriors.setValue(false);
                filename = "n" + Integer.toString(burnIn) + "_m" + Double.toString(magnitude) + "_posterior";
                dataStream.prepareForUse();
                dataStream.writeDataStreamToFile(folder + filename + extension);

                dataStream.driftConditional.setValue(true);
                dataStream.driftPriors.setValue(true);
                filename = "n" + Integer.toString(burnIn) + "_m" + Double.toString(magnitude) + "_both";
                dataStream.prepareForUse();
                dataStream.writeDataStreamToFile(folder + filename + extension);
            }
        }
    }
}
