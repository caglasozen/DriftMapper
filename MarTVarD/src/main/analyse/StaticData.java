package main.analyse;

import main.models.DriftMeasurement;
import main.models.Model;
import main.models.frequency.FrequencyMaps;
import main.models.frequency.FrequencyTable;
import main.report.ExperimentResult;
import weka.core.Instances;

import java.util.*;

/**
 * Created by LoongKuan on 31/07/2016.
 **/
public class StaticData {
    private Map<int[], ExperimentResult> resultMap;

    public StaticData(Instances instances1, Instances instances2,
                      int nAttributesActive, int[] attributeIndices, DriftMeasurement driftMeasurement) {
        this(instances1, instances2, nAttributesActive, attributeIndices, -1, 1, driftMeasurement, 0);
    }

    public StaticData(Instances instances1, Instances instances2,
                      int nAttributesActive, int[] attributeIndices,
                      double sampleScale, int nTests, DriftMeasurement driftMeasurement, int model) {
        // Generate base models for each data set
        // TODO : Change this
        Model model1;
        Model model2;
        if (model == 0) {
            model1 = new FrequencyTable(instances1, nAttributesActive, attributeIndices);
            model2 = new FrequencyTable(instances2, nAttributesActive, attributeIndices);
            model1.addInstances(instances1);
            model2.addInstances(instances2);
        }
        else {
            model1 = new FrequencyMaps(instances1, nAttributesActive, attributeIndices);
            model2 = new FrequencyMaps(instances2, nAttributesActive, attributeIndices);
            model1.addInstances(instances1);
            model2.addInstances(instances2);
        }

        this.resultMap = model1.analyseDifference(model2, sampleScale, nTests, driftMeasurement);
    }

    public Map<int[], ExperimentResult> getResultMap() {
        return resultMap;
    }
}
