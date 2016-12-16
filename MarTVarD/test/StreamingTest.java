import com.sun.corba.se.impl.protocol.INSServerRequestDispatcher;
import main.DriftMeasurement;
import main.analyse.streaming.MovingBase;
import main.analyse.streaming.NaiveWindowCompare;
import main.analyse.streaming.StaticBase;
import main.models.Model;
import main.models.frequency.FrequencyMaps;
import moa.classifiers.core.driftdetection.ADWINChangeDetector;
import weka.core.Instances;

import java.io.File;
import java.util.ArrayList;

/**
 * Created by loongkuan on 12/12/2016.
 */
public class StreamingTest extends MainTest{

    // TODO : Allow different window size and drift type and name result file accordingly
    public static void main(String[] args) {
        //String folder = "synthetic_5Att_5Val";
        //args = new String[]{"stream", "n1000000_m0.7_posterior"};
        //args = new String[]{"stream", "synthetic_5Att_5Val/n1000000_none"};

        //String folder = "data_uni_antwerp";
        //args = new String[]{"stream", "water_2016"};

        String folder = "";
        args = new String[]{"stream", "elecNormNew"};

        //String folder = "train_seed";
        //args = new String[]{"all", "20130419", "20131129"};

        //Instances[] dataSets = loadPairData(args[1], args[2]);
        //Instances allInstances = loadAnyDataSet("./datasets/" + folder + "/" + args[1] + ".arff");
        //allInstances.addAll(loadAnyDataSet("./datasets/" + folder + "/" + args[2] + ".arff"));
        //args[1] = args[1] + "_" + args[2];
        Instances allInstances = loadAnyDataSet("./datasets/" + folder + "/" + args[1] + ".arff");
        String filepath = getFilePath("./data_out", folder, args[1], "stream");

        //TODO: Be able to measure different types of drift and name accordingly
        //testAllWindowSize(allInstances, new int[]{100, 500, 1000, 5000, 10000}, filepath);
        testAllWindowSizeSubsetLength(allInstances,
                getAllWindowSize(allInstances), getAllAttributeSubsetLength(allInstances), filepath);
    }

    private static int[] getAllWindowSize(Instances instances) {
        int currentSize = 1000;
        ArrayList<Integer> allSizes  = new ArrayList<>();
        while (currentSize < instances.size()) {
            allSizes.add(currentSize);
            currentSize = Integer.toString(currentSize).charAt(0) == '1' ? currentSize * 5 : currentSize * 2;
        }
        int[] returnSizes = new int[allSizes.size()];
        for(int i = 0; i < allSizes.size(); i++) returnSizes[i] = allSizes.get(i);
        return returnSizes;
    }

    private static int[] getAllAttributeSubsetLength(Instances instances) {
        int maxLength = Math.min(instances.numAttributes(), 3);
        int[] allLength = new int[maxLength];
        for (int i = 0; i < maxLength; i++) {
            allLength[i] = i + 1;
        }
        return allLength;
    }

    private static void testAllWindowSizeSubsetLength(Instances instances,
                                                      int[] windowSizes, int[] subsetLength, String folder) {
        for (int size : windowSizes) {
            for (int length : subsetLength) {
                for (DriftMeasurement type : DriftMeasurement.values()) {
                    String file = folder + "/" + type.name() + "_" + size + "_" + length + ".csv";
                    testOnData(instances, length, size, type, file);
                }
            }
        }
    }

    // TODO: Automate testing with different windows
    private static void testOnData(Instances instances, int attributeSubsetLength, int windowSize,
                                   DriftMeasurement driftType, String resultFile) {
        int[] attributeIndices = new int[instances.numAttributes() - 1];
        for (int i = 0; i < instances.numAttributes() - 1; i++) attributeIndices[i] = i;

        Model referenceModel = new FrequencyMaps(instances, attributeSubsetLength, attributeIndices);
        NaiveWindowCompare streamingData = new NaiveWindowCompare(windowSize, referenceModel, driftType);

        int percentage = 0;
        for (int i = 0; i < instances.size(); i++) {
            if (percentage != (i * 100) / instances.size() ) {
                percentage = (i * 100 )/ instances.size();
                System.out.print("\r" + percentage + "% of instance processed");
            }
            streamingData.addInstance(instances.get(i));
        }
        System.out.println("\rDone test of Window Size = " + windowSize +
                ", Subset Length = " + attributeSubsetLength +
                ", and Drift Type = " + driftType.name());
        streamingData.writeResultsToFile(resultFile);
    }

    private static String getFilePath(String resultDir, String dataDir, String dataFileName, String experimentName) {
        String filname = resultDir;
        new File(filname).mkdir();
        filname += dataDir.equals("") ? "" : "/"  + dataDir;
        new File(filname).mkdir();
        filname += "/" + dataFileName;
        new File(filname).mkdir();
        filname += "/" + experimentName;
        new File(filname).mkdir();
        return filname;
    }
}
