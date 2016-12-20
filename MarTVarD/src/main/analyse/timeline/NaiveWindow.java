package main.analyse.timeline;

import main.DriftMeasurement;
import main.models.Model;
import main.models.frequency.FrequencyMaps;
import weka.core.Instance;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by loongkuan on 14/12/2016.
 */
public class NaiveWindow extends TimelineAnalysis{
    //TODO: Be able to measure different types of drift and name accordingly
    private int windowSize;

    public NaiveWindow(int windowSize, Model referenceModel, DriftMeasurement[] driftMeasurementType) {
        this.windowSize = windowSize;
        this.driftMeasurementTypes = driftMeasurementType;
        this.currentModel = referenceModel.copy();
        this.currentAllModel = referenceModel.copy();
        ((FrequencyMaps)currentAllModel).changeAttributeSubsetLength(this.currentModel.getAllInstances().numAttributes() - 1);
        this.attributeSubsets = referenceModel.getAllAttributeSubsets();

        this.currentIndex = -1;

        this.driftPoints = new HashMap<>();
        this.driftValues = new HashMap<>();
        for (DriftMeasurement type : this.driftMeasurementTypes) {
            this.driftPoints.put(type, new ArrayList<>());
            this.driftValues.put(type, new ArrayList<>());
        }
    }

    public void addInstance(Instance instance) {
        this.currentIndex += 1;
        if (currentModel.size() < this.windowSize) {
            this.currentModel.addInstance(instance);
            this.currentAllModel.addInstance(instance);
        }
        else {
            if (this.previousModel != null) {
                this.addDistance(this.currentIndex - this.windowSize);
            }
            this.previousModel = this.currentModel;
            this.currentModel = this.currentModel.copy();
            this.currentModel.addInstance(instance);

            this.previousAllModel = this.currentAllModel;
            this.currentAllModel= this.currentAllModel.copy();
            this.currentAllModel.addInstance(instance);
        }
    }
}
