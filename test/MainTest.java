import com.opencsv.CSVWriter;
import main.analyse.StaticData;
import main.DriftMeasurement;
import main.report.SummaryReport;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

import java.io.*;
import java.util.ArrayList;

/**
 * Created by LoongKuan on 28/08/2016.
 **/

public class MainTest {
    public static void main(String[] args) {
        String folder = "data_uni_antwerp";
        args = new String[]{"pair", "water_2015", "water_2016"};
        String[] standardFiles = new String[]{"water_2015"};
        //String[] standardFiles = new String[]{"gas-sensor"};
        //String[] standardFiles = new String[]{"n1000000_m0.7_posterior"};
        //String folder = "synthetic_5Att_5Val";
        //args = new String[]{"self"};
        double sampleScale = 1.0;
        int nTests = 10;
        //String folder = "train_seed";
        //args = new String[]{"all", "20130419", "20131129"};
        //args = new String[]{"pair", "20130505", "20131129"};
        //args = new String[]{"all", "20130606", "20131129"};
        //args = new String[]{"all", "20130708", "20131129"};
        //args = new String[]{"all", "20130910", "20131129"};
        //args = new String[]{"all", "20131113", "20131129"};
        //args = new String[]{"all", "20131129", "20131129"};
        //args = new String[]{"priorTest", "elecNormNew"};
        //args = new String[]{"testAll"};
        if (args[0].equals("self")) {
            compareSelf(new int[]{1, 2, 3, -1}, standardFiles, folder, sampleScale);
        }
        else if (args[0].equals("pair")) {
            comparePairs(new int[]{1, 2, 3, -1}, args[1], args[2], folder, 1.0);
        }
        else if (args[0].equals("testAll")) {
            testAllSatellite();
        }
    }

    private static void testAllSatellite() {
        String[] satelliteFiles = new String[]{"20130419", "20130505", "20130521", "20130606", "20130622",
                "20130708", "20130724", "20130809", "20130825", "20130910",
                "20130926", "20131012", "20131028", "20131113", "20131129"};

        for (int i = 0; i < satelliteFiles.length - 1; i++) {
            comparePairs(new int[]{1, 2, 3, -1},
                    satelliteFiles[i], satelliteFiles[i+1], "train_seed", 1.0);
        }
    }

    public static void compareSelf(int[] attributeSubsetLengths, String[] files, String folder, double sampleScale) {
        String folder1 = folder.equals("") ? "./datasets/" : "./datasets/" + folder + "/";
        for (String file : files) {
            Instances allInstances = loadAnyDataSet(folder1 + file +".arff");
            attributeSubsetLengths = getMaxSubsetLength(attributeSubsetLengths, allInstances);
            Instances[] dataSet = new Instances[2];
            dataSet[0] = new Instances(allInstances, 0, allInstances.size()/2);
            dataSet[1] = new Instances(allInstances, allInstances.size()/2 - 1, allInstances.size()/2);
            testAll(attributeSubsetLengths, dataSet, file, folder, sampleScale);
        }
    }

    private static void comparePairs(int[] attributeSubsetLengths, String file1, String file2,
                                    String folder, double sampleScale) {
        String dataFolder = folder.equals("") ? "./datasets/" : "./datasets/" + folder + "/";
        Instances instances1 = loadAnyDataSet(dataFolder + file1 + ".arff");
        Instances instances2 = loadAnyDataSet(dataFolder + file2 + ".arff");
        attributeSubsetLengths = getMaxSubsetLength(attributeSubsetLengths, instances1);
        testAll(attributeSubsetLengths, new Instances[]{instances1, instances2},
                file1 + "_" + file2, folder, sampleScale);
    }

    private static int[] getMaxSubsetLength(int[] attributeSubsetLengths, Instances dataset) {
        for (int i = 0; i < attributeSubsetLengths.length; i++) {
            if (attributeSubsetLengths[i] == -1) {
                attributeSubsetLengths[i] = dataset.numAttributes() - 1;
            }
        }
        return attributeSubsetLengths;
    }

    private static void testAll(int[] attributeSubsetLengths, Instances[] dataSets, String name, String folder, double sampleScale) {
        System.out.println("Running Tests on " + name);
        System.out.println("For " + attributeSubsetLengths[0] + " to " + attributeSubsetLengths[1] + " attributes");

        String rootDir = "./data_out/";
        new File(rootDir).mkdir();
        new File(rootDir + folder).mkdir();

        int model = 1;
        folder += "/" + name;
        new File(rootDir + folder).mkdir();

        folder += model == 0 ? "/FrequencyTable" : "/FrequencyMaps";
        new File(rootDir + folder).mkdir();

        folder = sampleScale <= 1 ? folder : folder + "/SampleSscale_" + sampleScale;
        new File(rootDir + folder).mkdir();

        int[] attributeIndices = new int[dataSets[0].numAttributes() - 1];
        for (int i = 0; i < dataSets[0].numAttributes() - 1; i++) attributeIndices[i] = i;

        for (int i : attributeSubsetLengths) {
            System.out.println("Running Covariate, Joint, Likelihood, and Posterior (in order) with subset length " + i);
            StaticData experiment = new StaticData(dataSets[0], dataSets[1], i, attributeIndices,
                    sampleScale, 10, DriftMeasurement.COVARIATE, model);
            SummaryReport summaryReport = new SummaryReport(experiment.getResultMap(), true);
            String filepath = rootDir + folder + "/" + name + "_" + i + "-attributes_covariate.csv";
            summaryReport.writeToCsv(filepath);

            experiment = new StaticData(dataSets[0], dataSets[1], i, attributeIndices,
                    sampleScale, 10, DriftMeasurement.JOINT, model);
            summaryReport = new SummaryReport(experiment.getResultMap(), true);
            filepath = rootDir + folder + "/" + name + "_" + i + "-attributes_joint.csv";
            summaryReport.writeToCsv(filepath);

            experiment = new StaticData(dataSets[0], dataSets[1], i, attributeIndices,
                    sampleScale, 10, DriftMeasurement.LIKELIHOOD, model);
            summaryReport = new SummaryReport(experiment.getResultMap(), true);
            filepath = rootDir + folder + "/" + name + "_" + i + "-attributes_likelihood.csv";
            summaryReport.writeToCsv(filepath);
            summaryReport = new SummaryReport(experiment.getResultMap(), false);
            filepath = rootDir + folder + "/" + name + "_" + i + "-attributes_likelihood_detailed.csv";
            summaryReport.writeToCsv(filepath);

            experiment = new StaticData(dataSets[0], dataSets[1], i, attributeIndices,
                    sampleScale, 10, DriftMeasurement.POSTERIOR, model);
            summaryReport = new SummaryReport(experiment.getResultMap(), true);
            filepath = rootDir + folder + "/" + name + "_" + i + "-attributes_posterior.csv";
            summaryReport.writeToCsv(filepath);
            summaryReport = new SummaryReport(experiment.getResultMap(), false);
            filepath = rootDir + folder + "/" + name + "_" + i + "-attributes_posterior_detailed.csv";
            summaryReport.writeToCsv(filepath);
        }
    }

    static void writeToCSV(String[][] data, String[] header, String filename){
        try {
            CSVWriter writer = new CSVWriter(new FileWriter(filename), ',');
            // feed in your array (or convert your data to an array)
            writer.writeNext(header);
            for (String[] dataLine : data) {
                writer.writeNext(dataLine);
            }
            writer.close();
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    static Instances loadDataSet(String filename) throws IOException {
        // Check if any attribute is numeric
        Instances result;
        BufferedReader reader;

        reader = new BufferedReader(new FileReader(filename));
        result = new Instances(reader);
        reader.close();
        return result;
    }

    private static Instances discretizeDataSet(Instances dataSet) throws Exception{
        ArrayList<Integer> continuousIndex = new ArrayList<>();
        for (int i = 0; i < dataSet.numAttributes(); i++) {
            if (dataSet.attribute(i).isNumeric()) continuousIndex.add(i);
        }
        int[] attIndex = new int[continuousIndex.size()];
        for (int i = 0; i < continuousIndex.size(); i++) {
            attIndex[i] = continuousIndex.get(i);
        }

        Discretize filter = new Discretize();
        filter.setUseEqualFrequency(true);
        filter.setBins(5);
        filter.setAttributeIndicesArray(attIndex);
        filter.setInputFormat(dataSet);

        return Filter.useFilter(dataSet, filter);
    }

    public static Instances loadAnyDataSet(String filename) {
        try {
            Instances continuousData = loadDataSet(filename);
            if (filename.equals("./datasets/gas-sensor.arff")) {
                double[] classAttVals = continuousData.attributeToDoubleArray(0);
                Attribute classAtt = continuousData.attribute(0);
                continuousData.deleteAttributeAt(0);
                continuousData.insertAttributeAt(classAtt, continuousData.numAttributes());
                for (int i = 0; i < classAttVals.length; i++) {
                    continuousData.get(i).setValue(continuousData.classIndex(), classAttVals[i]);
                }
            }
            Instances instances = discretizeDataSet(continuousData);
            instances.setClassIndex(instances.numAttributes() - 1);
            return instances;
        }
        catch (Exception ex) {
            ex.printStackTrace();
            return new Instances("E", new ArrayList<Attribute>(), 0);
        }
    }
}
