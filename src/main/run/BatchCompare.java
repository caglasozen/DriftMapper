package main.run;

import main.analyse.StaticData;
import main.report.SummaryReport;
import org.apache.commons.lang3.ArrayUtils;
import weka.core.Instances;
import main.DriftMeasurement;

import java.util.ArrayList;

/**
 * Created by loongkuan on 16/12/2016.
 */
public class BatchCompare extends main{

    public static void BatchCompare(String resultFolder, Instances instances, int[] splitIndices, int[] subsetLengths) {
        //int[] subsetLengths = getAllAttributeSubsetLength(instances);
        subsetLengths = ArrayUtils.add(subsetLengths, instances.numAttributes() - 1);

        Instances[] instancesPair = new Instances[2];
        //TODO: Check math of making window size from indices
        instancesPair[0] = new Instances(instances, splitIndices[0], splitIndices[1] - splitIndices[0]);
        instancesPair[1] = new Instances(instances, splitIndices[2], splitIndices[3] - splitIndices[2]);
        runExperiment(subsetLengths, instancesPair, resultFolder, 1.0);
    }

    public static void compareSelf(int[] attributeSubsetLengths, String[] files, String folder, double sampleScale) {
        String folder1 = folder.equals("") ? "./datasets/" : "./datasets/" + folder + "/";
        for (String file : files) {
            Instances allInstances = loadAnyDataSet(folder1 + file +".arff");
            attributeSubsetLengths = getMaxSubsetLength(attributeSubsetLengths, allInstances);
            Instances[] dataSet = new Instances[2];
            dataSet[0] = new Instances(allInstances, 0, allInstances.size()/2);
            dataSet[1] = new Instances(allInstances, allInstances.size()/2 - 1, allInstances.size()/2);
            String resultFolder = main.createFilePath(new String[]{"./data_out", "folder", file, "FrequencyMap"});
            runExperiment(attributeSubsetLengths, dataSet, resultFolder, sampleScale);
        }
    }

    private static void comparePairs(int[] attributeSubsetLengths, String file1, String file2,
                                    String folder, double sampleScale) {
        String dataFolder = folder.equals("") ? "./datasets/" : "./datasets/" + folder + "/";
        Instances instances1 = loadAnyDataSet(dataFolder + file1 + ".arff");
        Instances instances2 = loadAnyDataSet(dataFolder + file2 + ".arff");
        attributeSubsetLengths = getMaxSubsetLength(attributeSubsetLengths, instances1);
        String resultFolder = main.createFilePath(new String[]{"./data_out", "folder", file1 + "_" + file2, "FrequencyMap"});
        runExperiment(attributeSubsetLengths, new Instances[]{instances1, instances2}, resultFolder, sampleScale);
    }

    private static int[] getMaxSubsetLength(int[] attributeSubsetLengths, Instances dataset) {
        for (int i = 0; i < attributeSubsetLengths.length; i++) {
            if (attributeSubsetLengths[i] == -1) {
                attributeSubsetLengths[i] = dataset.numAttributes() - 1;
            }
        }
        return attributeSubsetLengths;
    }

    private static void runExperiment(int[] attributeSubsetLengths, Instances[] dataSets, String resultFolder, double sampleScale) {
        String[] folders = resultFolder.split("/");
        System.out.println("Running Tests on " + folders[folders.length - 2]);
        System.out.println("For " + attributeSubsetLengths[0] + " to " + attributeSubsetLengths[1] + " attributes");

        int model = 1;

        int[] attributeIndices = getAttributeIndicies(dataSets[0]);

        for (int i : attributeSubsetLengths) {
            System.out.println("Running Covariate, Joint, Likelihood, and Posterior (in order) with subset length " + i);
            StaticData experiment = new StaticData(dataSets[0], dataSets[1], i, attributeIndices,
                    sampleScale, 10, DriftMeasurement.COVARIATE, model);
            SummaryReport summaryReport = new SummaryReport(experiment.getResultMap(), true);
            String filepath = resultFolder + "/" + i + "-attributes_covariate.csv";
            summaryReport.writeToCsv(filepath);

            experiment = new StaticData(dataSets[0], dataSets[1], i, attributeIndices,
                    sampleScale, 10, DriftMeasurement.JOINT, model);
            summaryReport = new SummaryReport(experiment.getResultMap(), true);
            filepath = resultFolder + "/" + i + "-attributes_joint.csv";
            summaryReport.writeToCsv(filepath);

            experiment = new StaticData(dataSets[0], dataSets[1], i, attributeIndices,
                    sampleScale, 10, DriftMeasurement.LIKELIHOOD, model);
            summaryReport = new SummaryReport(experiment.getResultMap(), true);
            filepath = resultFolder + "/" + i + "-attributes_likelihood.csv";
            summaryReport.writeToCsv(filepath);
            summaryReport = new SummaryReport(experiment.getResultMap(), false);
            filepath = resultFolder + "/" + i + "-attributes_likelihood_detailed.csv";
            summaryReport.writeToCsv(filepath);

            experiment = new StaticData(dataSets[0], dataSets[1], i, attributeIndices,
                    sampleScale, 10, DriftMeasurement.POSTERIOR, model);
            summaryReport = new SummaryReport(experiment.getResultMap(), true);
            filepath = resultFolder + "/" + i + "-attributes_posterior.csv";
            summaryReport.writeToCsv(filepath);
            summaryReport = new SummaryReport(experiment.getResultMap(), false);
            filepath = resultFolder + "/" + i + "-attributes_posterior_detailed.csv";
            summaryReport.writeToCsv(filepath);
        }
    }
}
