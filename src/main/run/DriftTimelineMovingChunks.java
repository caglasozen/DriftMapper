package main.run;

import main.DriftMeasurement;
import main.analyse.timeline.NaiveChunk;
import main.analyse.timeline.NaiveMovingChunk;
import main.models.Model;
import main.models.frequency.FrequencyMaps;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Created by loongkuan on 20/12/2016.
 */
public class DriftTimelineMovingChunks extends DriftTimelineChunks{

    //TODO: Bug due to discretizing date attributes
    public static void DriftTimelineMovingChunks(String resultFolder, Instances allInstances,
                                           int groupAttribute, int[] groupSizes, int[] subsetLengths) {
        Instances[] allGroupedInstances;
        if (groupAttribute >= 0) {
            ArrayList<Instances> groupedInstances = new ArrayList<>();
            double prevAttributeValue = -1;
            double[] groupedAttVals = allInstances.attributeToDoubleArray(groupAttribute);
            allInstances = discretizeDataSet(allInstances);
            allInstances.deleteAttributeAt(groupAttribute);
            for (int j = 0; j < allInstances.size(); j++) {
                Instance instance = allInstances.get(j);
                if (groupedAttVals[j] != prevAttributeValue) {
                    groupedInstances.add(new Instances(allInstances, 0));
                    prevAttributeValue = groupedAttVals[j];
                }
                groupedInstances.get(groupedInstances.size() - 1).add(instance);
            }
            allGroupedInstances = groupedInstances.toArray(new Instances[groupedInstances.size()]);
        }
        else {
            allGroupedInstances = new Instances[]{discretizeDataSet(allInstances)};
        }

        for (int subsetLength : subsetLengths) {
            for (int groupSize : groupSizes) {
                runExperiment(allGroupedInstances, subsetLength, groupSize, resultFolder);
            }
        }
    }

    private static void runExperiment(Instances[] allInstances, int subsetLength, int groupSize, String resultFolder) {
        int[] attributeIndices = getAttributeIndicies(allInstances[0]);

        Model referenceModel = new FrequencyMaps(allInstances[0], subsetLength, attributeIndices);
        NaiveMovingChunk naiveChunk = new NaiveMovingChunk(allInstances, groupSize, DriftMeasurement.values(), referenceModel);

        System.out.println("\rDone test of Subset Length = " + subsetLength + " and chunk size of " + groupSize);

        for (DriftMeasurement driftMeasurement : DriftMeasurement.values()) {
            String file = resultFolder + "/" + driftMeasurement.name() + "_" + groupSize + "_" + subsetLength + ".csv";
            naiveChunk.writeResultsToFile(file, driftMeasurement);
        }
    }
}
