package analyse.timeline;

import global.DriftMeasurement;
import models.Model;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;

/**
 * Created by loongkuan on 20/12/2016.
 */
public class NaiveMovingChunk extends TimelineAnalysis{
    private Queue<Integer> chunkIndex1;
    private Queue<Integer> chunkIndex2;

    public NaiveMovingChunk(Instances[] allInstances, int chunkSize, DriftMeasurement[] driftTypes, Model referenceModel) {
        this.previousModel = referenceModel.copy();
        this.currentModel = referenceModel.copy();

        this.currentIndex = 0;
        this.driftMeasurementTypes = driftTypes;

        this.attributeSubsets = this.previousModel.getAllAttributeSubsets();

        this.driftPoints = new HashMap<>();
        this.driftValues = new HashMap<>();
        for (DriftMeasurement type : this.driftMeasurementTypes) {
            this.driftPoints.put(type, new ArrayList<>());
            this.driftValues.put(type, new ArrayList<>());
        }

        this.chunkIndex1 = new LinkedList<>();
        this.chunkIndex2 = new LinkedList<>();

        for (int i = 0; i < allInstances.length; i++) {
            System.out.print("\rProcessing chunk: " + (i + 1) + "/" + allInstances.length);
            if (chunkIndex1.size() < chunkSize) {
                this.previousModel.addInstances(allInstances[i]);
                this.chunkIndex1.add(i);
                this.currentIndex += allInstances[i].size();
            }
            else if (chunkIndex2.size() < chunkSize) {
                this.currentModel.addInstances(allInstances[i]);
                this.chunkIndex2.add(i);
            }
            else {
                // Move Window
                // Remove tail chunk
                int tailChunkIndex = this.chunkIndex1.remove();
                this.previousModel.removeInstances(this.intArraySequence(allInstances[tailChunkIndex].size()));
                // Move middle chunk
                int middleChunkIndex = this.chunkIndex2.remove();
                this.currentModel.removeInstances(this.intArraySequence(allInstances[middleChunkIndex].size()));
                this.previousModel.addInstances(allInstances[middleChunkIndex]);
                this.chunkIndex1.add(middleChunkIndex);
                // Add head chunk
                this.currentModel.addInstances(allInstances[i]);
                this.chunkIndex2.add(i);
                // Increment middle index point tracker
                this.currentIndex += allInstances[middleChunkIndex].size();
            }
            if (chunkIndex1.size() == chunkSize && chunkIndex2.size() == chunkSize){
                this.addDistance(this.currentIndex);
            }
        }
    }

    private int[] intArraySequence(int n) {
        // Creates array of int from 0 to n-1
        int[] array = new int[n];
        for (int i = 0; i < n; i++) {
            array[i] = i;
        }
        return array;
    }

     @Override
     public void addInstance(Instance instance) {
     }
}
