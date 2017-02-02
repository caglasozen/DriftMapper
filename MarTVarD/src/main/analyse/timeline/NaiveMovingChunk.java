package main.analyse.timeline;

import main.DriftMeasurement;
import main.models.Model;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by loongkuan on 20/12/2016.
 */
public class NaiveMovingChunk extends TimelineAnalysis{
    private ArrayList<Integer> chunkSizes;
    private double previousGroupValue;

    public NaiveMovingChunk(int groupAttributeIndex, DriftMeasurement[] driftTypes, Model referenceModel) {
        this.previousModel = referenceModel.copy();

        this.currentIndex = -1;
        this.driftMeasurementTypes = driftTypes;

        this.attributeSubsets = this.previousModel.getAllAttributeSubsets();

        this.driftPoints = new HashMap<>();
        this.driftValues = new HashMap<>();
        for (DriftMeasurement type : this.driftMeasurementTypes) {
            this.driftPoints.put(type, new ArrayList<>());
            this.driftValues.put(type, new ArrayList<>());
        }

        this.chunkSizes = new ArrayList<>();
        this.previousGroupValue = -1;

        for (int i = 1; i < allInstances.length; i++) {
            System.out.print("\rProcessing chunk: " + (i + 1) + "/" + allInstances.length);
            this.currentModel = previousModel.copy();
            this.currentModel.addInstances(allInstances[i]);

            this.addDistance(this.currentIndex);

            this.currentIndex += allInstances[i].size();
            this.previousModel = this.currentModel;
        }
    }

     @Override
     public void addInstance(Instance instance) {
     }
}
