import main.analyse.streaming.MovingBase;
import main.analyse.streaming.StaticBase;
import main.models.Model;
import main.models.frequency.FrequencyMaps;
import moa.classifiers.core.driftdetection.ADWINChangeDetector;
import weka.core.Instances;

/**
 * Created by loongkuan on 12/12/2016.
 */
public class StreamingTest extends MainTest{

    public static void main(String[] args) {
        args = new String[]{"stream", "gas-sensor"};

        //Instances[] dataSets = loadPairData(args[1], args[2]);
        //Instances allInstances = dataSets[0];
        //allInstances.addAll(dataSets[1]);
        Instances allInstances = loadAnyDataSet("./datasets/" + args[1] + ".arff");
        int[] attributeIndices = new int[allInstances.numAttributes() - 1];
        for (int i = 0; i < allInstances.numAttributes() - 1; i++) attributeIndices[i] = i;
        Model model = new FrequencyMaps(allInstances, allInstances.numAttributes() - 1, attributeIndices);
        MovingBase streamingData = new MovingBase(model);
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
        duration = (System.currentTimeMillis() - startTime) / 1000;
        System.out.println("Time taken: " + duration);
        streamingData.printDriftPoints();
    }

    private void benchmark() {
        ADWINChangeDetector detector = new ADWINChangeDetector();
        detector.prepareForUse();
    }
}
