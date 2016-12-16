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

        String folder = "data_uni_antwerp";
        args = new String[]{"stream", "water_2015"};

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
        testAllWindowSize(allInstances, new int[]{5000, 10000, 50000, 100000, 500000}, filepath);
    }

    private static void testAllWindowSize(Instances instances, int[] windowSizes, String folder) {
        ArrayList<int[]> attributesubsets = getAllAttributeSubsets(instances, 1);
        for (int size : windowSizes) {
            for (DriftMeasurement type : DriftMeasurement.values()) {
                String file = folder + "/" + type.name() + "_" + size + ".csv";
                testOnData(instances, attributesubsets, size, type, file);
            }
        }
    }

    private static ArrayList<int[]> getAllAttributeSubsets(Instances instances, int length) {
        ArrayList<int[]> subsets = new ArrayList<>();
        for (int i = 0; i < instances.numAttributes() - 1; i++) {
            subsets.add(new int[]{i});
        }
        return subsets;
    }

    // TODO: Automate testing with different windows
    private static void testOnData(Instances instances, ArrayList<int[]> attributeSubsets, int windowSize,
                                   DriftMeasurement driftType, String resultFile) {
        int[] attributeIndices = new int[instances.numAttributes() - 1];
        for (int i = 0; i < instances.numAttributes() - 1; i++) attributeIndices[i] = i;
        Model model = new FrequencyMaps(instances, attributeSubsets.get(0).length, attributeIndices);
        NaiveWindowCompare streamingData = new NaiveWindowCompare(windowSize, model, attributeSubsets, driftType);
        long startTime = System.currentTimeMillis();
        System.out.println("");
        long duration = 0;
        for (int i = 0; i < instances.size(); i++) {
            streamingData.addInstance(instances.get(i));
            if (duration != (System.currentTimeMillis() - startTime) / 1000) {
                duration = (System.currentTimeMillis() - startTime) / 1000;
                System.out.print("\rAdded " + i + " Instances out of " + instances.size() +
                        " at " + i / duration + " instances per second");
            }
        }
        System.out.println("");
        duration = (System.currentTimeMillis() - startTime) / 1000;
        System.out.println("Time taken: " + duration);
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

    private void benchmark() {
        ADWINChangeDetector detector = new ADWINChangeDetector();
        detector.prepareForUse();
    }
}
