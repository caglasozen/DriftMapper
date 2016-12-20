package main.analyse.timeline;

import main.DriftMeasurement;
import main.models.Model;
import main.models.frequency.FrequencyMaps;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by loongkuan on 20/12/2016.
 */
public class NaiveChunks extends TimelineAnalysis{

    public NaiveChunks(Instances[] allInstances, DriftMeasurement[] driftTypes, Model referenceModel) {
        this.previousAllModel = referenceModel.copy();
        ((FrequencyMaps)previousAllModel).changeAttributeSubsetLength(allInstances[0].numAttributes() - 1);
        this.previousAllModel.addInstances(allInstances[0]);
        this.previousModel = referenceModel.copy();
        this.previousModel.addInstances(allInstances[0]);

        this.currentIndex = allInstances[0].size();
        this.driftMeasurementTypes = driftTypes;

        this.attributeSubsets = this.previousModel.getAllAttributeSubsets();

        this.driftPoints = new HashMap<>();
        this.driftValues = new HashMap<>();
        for (DriftMeasurement type : this.driftMeasurementTypes) {
            this.driftPoints.put(type, new ArrayList<>());
            this.driftValues.put(type, new ArrayList<>());
        }

        for (int i = 1; i < allInstances.length; i++) {
            System.out.print("\rProcessing chunk: " + (i + 1) + "/" + allInstances.length);
            this.currentAllModel = previousAllModel.copy();
            this.currentAllModel.addInstances(allInstances[i]);
            this.currentModel = previousModel.copy();
            this.currentModel.addInstances(allInstances[i]);

            this.addDistance(this.currentIndex);

            this.currentIndex += allInstances[i].size();
            this.previousAllModel = this.currentAllModel;
            this.previousModel = this.currentModel;
        }
    }
}
