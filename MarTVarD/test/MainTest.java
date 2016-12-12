import com.opencsv.CSVWriter;
import main.analyse.StaticData;
import main.analyse.streaming.MovingBase;
import main.analyse.streaming.StaticBase;
import main.models.DriftMeasurement;
import main.models.Model;
import main.models.frequency.FrequencyMaps;
import main.report.SummaryReport;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SystemInfo;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

import java.io.*;
import java.util.ArrayList;

/**
 * Created by LoongKuan on 28/08/2016.
 **/

public class MainTest {
    public static void main(String[] args) {
        //String[] standardFiles = new String[]{"airlines"};
        String[] standardFiles = new String[]{"elecNormNew", "sensor", "airlines"};
        //args = new String[]{"standardAll"};
        double sampleScale = 1.0;
        int nTests = 10;
        //args = new String[]{"stream", "20130505", "20131129"};
        //args = new String[]{"all", "20130419", "20131129"};
        //args = new String[]{"all", "20130505", "20131129"};
        //args = new String[]{"all", "20130606", "20131129"};
        //args = new String[]{"all", "20130708", "20131129"};
        //args = new String[]{"all", "20130910", "20131129"};
        //args = new String[]{"all", "20131113", "20131129"};
        //args = new String[]{"all", "20131129", "20131129"};
        //args = new String[]{"priorTest", "elecNormNew"};
        args = new String[]{"testAll"};
        if (args[0].equals("all")) {
            Instances[] dataSets = loadPairData(args[1], args[2]);
            int nAtt = dataSets[0].numAttributes();
            testAll(new int[]{1, 4}, dataSets, args[1] + "_" + args[2], sampleScale, "");
        }
        else if (args[0].equals("standardAll")) {
            standardAll(new int[]{1,4}, standardFiles, sampleScale, "");
        }
        else if (args[0].equals("testAll")) {
            testAllSatellite();
        }
        else if (args[0].equals("stream")) {
            Instances[] dataSets = loadPairData(args[1], args[2]);
            Instances allInstances = dataSets[1];
            int[] attributeIndices = new int[allInstances.numAttributes() - 1];
            for (int i = 0; i < allInstances.numAttributes() - 1; i++) attributeIndices[i] = i;
            Model model = new FrequencyMaps(allInstances, allInstances.numAttributes() - 1, attributeIndices);
            StaticBase streamingData = new StaticBase(dataSets[0], model, 2000);
            int percentage = -1;
            long startTime = System.currentTimeMillis();
            System.out.println("");
            long duration = 0;
            for (int i = 0; i < allInstances.size(); i++) {
                /*
                if (percentage != (int)((i/(double)allInstances.size()) * 100)) {
                    percentage = (int)((i/(double)allInstances.size()) * 100);
                    System.out.print("\rAdded " + percentage + "% of Instances ");
                }
                */
                streamingData.addInstance(allInstances.get(i));
                if (duration != (System.currentTimeMillis() - startTime) / 1000) {
                    duration = (System.currentTimeMillis() - startTime) / 1000;
                    System.out.print("\rAdded " + i + " Instances out of " + allInstances.size() +
                            " at " + i / duration + " instances per second");
                }
            }
            System.out.println("");
            streamingData.printDriftTimeLine();
        }
    }

    private static void testAllSatellite() {
        String[] satelliteFiles = new String[]{"20130419", "20130505", "20130521", "20130606", "20130622",
                "20130708", "20130724", "20130809", "20130825", "20130910",
                "20130926", "20131012", "20131028", "20131113", "20131129"};

        for (int i = 0; i < satelliteFiles.length - 1; i++) {
            Instances instances1 = loadSingleData(satelliteFiles[i]);
            Instances instances2 = loadSingleData(satelliteFiles[i+1]);
            testAll(new int[]{1, 4}, new Instances[]{instances1, instances2},
                    satelliteFiles[i] + "_" + satelliteFiles[i+1], 1.0, "");
            int numAtt = instances1.numAttributes();
            testAll(new int[]{numAtt - 1, numAtt}, new Instances[]{instances1, instances2},
                    satelliteFiles[i] + "_" + satelliteFiles[i+1], 1.0, "");
        }
    }

    public static void standardAll(int[] nInterval, String[] files, double sampleScale, String folder) {
        String folder1 = folder.equals("") ? "./datasets/" : "./datasets/" + folder + "/";
        String folder2 = folder.equals("") ? "martvard" : folder;
        for (String file : files) {
            Instances allInstances = loadAnyDataSet(folder1 + file +".arff");
            Instances[] dataSet = new Instances[2];
            dataSet[0] = new Instances(allInstances, 0, allInstances.size()/2);
            dataSet[1] = new Instances(allInstances, allInstances.size()/2 - 1, allInstances.size()/2);
            testAll(nInterval, dataSet, file, sampleScale, folder2);
        }
    }

    private static void testAll(int[] nInterval, Instances[] dataSets, String name, double sampleScale, String folder) {
        System.out.println("Running Tests on " + name);
        System.out.println("For " + nInterval[0] + " to " + nInterval[1] + " attributes");

        String rootDir = "./data_out/";
        new File(rootDir + folder).mkdir();

        int model = 1;
        folder += model == 0 ? "/FrequencyTable" : "/FrequencyMaps";
        new File(rootDir + folder).mkdir();

        folder = sampleScale <= 1 ? folder : folder + "/SampleSscale" + sampleScale;
        new File(rootDir + folder).mkdir();

        folder += "/" + name;
        new File(rootDir + folder).mkdir();

        int[] attributeIndices = new int[dataSets[0].numAttributes() - 1];
        for (int i = 0; i < dataSets[0].numAttributes() - 1; i++) attributeIndices[i] = i;

        String[] csvHeader = new String[]{"drift", "mean", "sd", "max_val", "max_att", "min_val", "min_att", "attributes"};

        for (int i = nInterval[0]; i < nInterval[1]; i++) {
            System.out.println("Running Covariate with subset length " + i);
            StaticData experiment = new StaticData(dataSets[0], dataSets[1], i, attributeIndices,
                    sampleScale, 10, DriftMeasurement.COVARIATE, model);
            SummaryReport summaryReport = new SummaryReport(experiment.getResultMap(), true);
            String filepath = "./data_out/" + folder + "/" + name + "_" + i + "-attributes_covariate.csv";
            summaryReport.writeToCsv(filepath);

            System.out.println("Running Joint with subset length " + i);
            experiment = new StaticData(dataSets[0], dataSets[1], i, attributeIndices,
                    sampleScale, 10, DriftMeasurement.JOINT, model);
            summaryReport = new SummaryReport(experiment.getResultMap(), true);
            filepath = "./data_out/" + folder + "/" + name + "_" + i + "-attributes_joint.csv";
            summaryReport.writeToCsv(filepath);

            System.out.println("Running Likelihood with subset length " + i);
            experiment = new StaticData(dataSets[0], dataSets[1], i, attributeIndices,
                    sampleScale, 10, DriftMeasurement.LIKELIHOOD, model);
            summaryReport = new SummaryReport(experiment.getResultMap(), true);
            filepath = "./data_out/" + folder + "/" + name + "_" + i + "-attributes_likelihood.csv";
            summaryReport.writeToCsv(filepath);
            summaryReport = new SummaryReport(experiment.getResultMap(), false);
            filepath = "./data_out/" + folder + "/" + name + "_" + i + "-attributes_likelihood_detailed.csv";
            summaryReport.writeToCsv(filepath);

            System.out.println("Running Posterior with subset length " + i);
            experiment = new StaticData(dataSets[0], dataSets[1], i, attributeIndices,
                    sampleScale, 10, DriftMeasurement.POSTERIOR, model);
            summaryReport = new SummaryReport(experiment.getResultMap(), true);
            filepath = "./data_out/" + folder + "/" + name + "_" + i + "-attributes_posterior.csv";
            summaryReport.writeToCsv(filepath);
            summaryReport = new SummaryReport(experiment.getResultMap(), false);
            filepath = "./data_out/" + folder + "/" + name + "_" + i + "-attributes_posterior_detailed.csv";
            summaryReport.writeToCsv(filepath);
        }
    }

    private static Instances[] loadPairData(String filename1, String filename2) {
        try {
            // Load data sets into collated data set and discretize
            Instances instances1 = loadDataSet("./datasets/train_seed/"+filename1+".arff");
            Instances instances2 = loadDataSet("./datasets/train_seed/"+filename2+".arff");
            Instances instances3 = new Instances(instances1);
            instances3.addAll(instances2);
            instances3 = discretizeDataSet(instances3);
            instances1 = new Instances(instances3, 0, instances1.size());
            instances2 = new Instances(instances3, instances1.size(), instances2.size());
            return new Instances[]{instances1, instances2};
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
        return new Instances[2];
    }

    private static Instances loadSingleData(String filename) {
        try {
            // Load data sets into collated data set and discretize
            Instances instances = loadDataSet("./datasets/train_seed/"+filename+".arff");
            instances = discretizeDataSet(instances);
            return instances;
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
        return null;
    }

    public static void writeToCSV(String[][] data, String[] header, String filename){
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

    private static Instances loadDataSet(String filename) throws IOException {
        // Check if any attribute is numeric
        Instances result;
        BufferedReader reader;

        reader = new BufferedReader(new FileReader(filename));
        result = new Instances(reader);
        result.setClassIndex(result.numAttributes() - 1);
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
                continuousData.setClassIndex(continuousData.numAttributes() - 1);
                for (int i = 0; i < classAttVals.length; i++) {
                    continuousData.get(i).setValue(continuousData.classIndex(), classAttVals[i]);
                }
            }
            return discretizeDataSet(continuousData);
        }
        catch (Exception ex) {
            ex.printStackTrace();
            return new Instances("E", new ArrayList<Attribute>(), 0);
        }
    }
}
