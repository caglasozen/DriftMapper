import main.analyse.StaticData;
import main.models.DriftMeasurement;
import main.report.SummaryReport;
import weka.core.Instances;

/**
 * Created by LoongKuan on 8/10/2016.
 */
public class SamplingTest {
    public static void main(String[] args) {
        int nAttributes = 5;
        int nValues = 5;
        double[] magnitudes = new double[]{0.5};
        int[] burnIns = new int[]{1000000};
        String[] type = new String[]{"prior", "posterior", "both"};

        String[] allFiles = new String[magnitudes.length * burnIns.length * type.length];

        for (int i = 0; i < magnitudes.length; i++) {
            for (int j = 0; j < burnIns.length; j++) {
                for (int k = 0; k < type.length; k++) {
                    int index = i * (burnIns.length * type.length) + j * type.length + k;
                    allFiles[index] = "n" + burnIns[j] + "_m" + magnitudes[i] + "_" + type[k];
                }
            }
        }

        String jointFile = "n1000000_m0.5_both";

        String folder = "synthetic_" + nAttributes + "Att_" + nValues + "Val";

        //MainTest.standardAll(new int[]{1,6}, allFiles, -1, folder);
        int nSizes = 60;
        int[] sampleSizes = new int[nSizes];
        for (int i = 0; i < nSizes; i++) {
            sampleSizes[i] = (i+1) * 100;
        }
        //MainTest.standardAll(new int[]{1,7}, allFiles, 1000, folder);
        JointTest(jointFile, folder, sampleSizes, true, 20);
        JointTest(jointFile, folder, sampleSizes, false, 20);
        PosteriorTest(jointFile, folder, sampleSizes, 20);
    }

    private static Instances[] GetSplitData(String filename, String folder) {
        String folder1 = folder.equals("") ? "./datasets/" : "./datasets/" + folder + "/";
        Instances allInstances = MainTest.loadAnyDataSet(folder1 + filename +".arff");
        Instances[] dataSet = new Instances[2];
        dataSet[0] = new Instances(allInstances, 0, allInstances.size()/2);
        dataSet[1] = new Instances(allInstances, allInstances.size()/2 - 1, allInstances.size()/2);
        return dataSet;
    }

    private static void JointTest(String filename, String folder, int[] sampleSizes, boolean joint, int nTests) {
        Instances[] dataSet = GetSplitData(filename, folder);

        int nActiveAttributes = joint ? dataSet[0].numAttributes() : dataSet[0].numAttributes() - 1;
        int[] attributeIndicesCov = new int[nActiveAttributes];
        for (int i = 0; i < nActiveAttributes; i++) attributeIndicesCov[i] = i;

        String[][] resultTable = new String[sampleSizes.length][10];
        for (int i = 0; i < sampleSizes.length; i++) {
            int sampleSize = sampleSizes[i];
            System.out.println("Running with sample size: " + sampleSize);
            resultTable[i][0] = Integer.toString(sampleSize);
            StaticData experiment = new StaticData(dataSet[0], dataSet[1], dataSet[0].numAttributes(), attributeIndicesCov, sampleSize, nTests, DriftMeasurement.COVARIATE, 0);
            String[] row = new SummaryReport(experiment.getResultMap(), true).getResultTable()[0];
            for (int j = 0; j < row.length; j++) {
                resultTable[i][j + 1] = row[j];
            }
        }
        filename = nTests > 0 ? filename + "_" + nTests + "averaged" : filename;
        String type = joint ? "JointTest" : "CovariateTest";
        MainTest.writeToCSV(resultTable,
                new String[]{"SampleSize", "Distance", "mean", "sd", "max_val", "max_att", "min_val", "min_att", "attributes", "class_values"},
                "./data_out/SampleSizeTest/" + filename + "_" + type + ".csv");
    }

    private static void PosteriorTest(String filename, String folder, int[] sampleSizes, int nTests) {
        Instances[] dataSet = GetSplitData(filename, folder);

        int[] attributeIndicesCov = new int[dataSet[0].numAttributes() - 1];
        for (int i = 0; i < dataSet[0].numAttributes() - 1; i++) attributeIndicesCov[i] = i;

        String[][] resultTable = new String[sampleSizes.length][10];
        for (int i = 0; i < sampleSizes.length; i++) {
            int sampleSize = sampleSizes[i];
            System.out.println("Running with sample size: " + sampleSize);
            resultTable[i][0] = Integer.toString(sampleSize);
            StaticData experiment = new StaticData(dataSet[0], dataSet[1], dataSet[0].numAttributes(), attributeIndicesCov, sampleSize, nTests, DriftMeasurement.POSTERIOR, 0);
            String[] row = new SummaryReport(experiment.getResultMap(), true).getResultTable()[0];
            for (int j = 0; j < row.length; j++) {
                resultTable[i][j + 1] = row[j];
            }
        }
        filename = nTests > 0 ? filename + "_" + nTests + "averaged" : filename;
        MainTest.writeToCSV(resultTable,
                new String[]{"SampleSize", "Distance", "mean", "sd", "max_val", "max_att", "min_val", "min_att", "attributes", "class_values"},
                "./data_out/SampleSizeTest/" + filename + "_PosteriorTest.csv");
    }
}
