package analyse.timeline;

import models.Model;
import org.apache.commons.lang3.ArrayUtils;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Created by LoongKuan on 5/07/2017.
 */
public class Windows extends TimelineAnalysis{
    private int windowSize;

    //TODO: add attributes to measure, instead of all attributes, maybe to listener
    public Windows(int windowSize, boolean increment, Model referenceModel) {
        this.previousModel = referenceModel.copy();
        this.currentModel = referenceModel.copy();

        this.currentIndex = 0;
        this.windowSize = windowSize;
        this.increment = increment;

        this.attributeSubsets = this.previousModel.getAllAttributeSubsets();

        this.updateListenerMetadata();
    }

    /**
     * Function to find the indices of the first window in the model
     * @param length The index of the attribute that the chunks are based off.
     * @return The list of increasing indices of the instances in the first chunk
     */
    private static int[] windowIndices(int length) {
        int[] indices = new int[length];
        for (int i = 0; i < length; i++) {
            indices[i] = i;
        }
        return indices;
    }

     @Override
     public void addInstance(Instance instance) {
        // Check if we are filling the tail/previous model
        if (this.previousModel.size() < this.windowSize) {
            this.previousModel.addInstance(instance);
            this.currentIndex += 1;
        }
        // Check if we are filling the head/current model
        else if (this.currentModel.size() < this.windowSize) {
            this.currentModel.addInstance(instance);
        }
        // Both models are filled with the max number of chunks
        else {
            // First calculate distance and call listeners
            this.addDistance(this.currentIndex);
            // If not analysis is not by increments, the operations are trivial
            if (!this.increment) {
                this.currentIndex += this.currentModel.size();
                this.previousModel = this.currentModel;
                this.currentModel = this.currentModel.copy();
                // Add instance
                this.currentModel.addInstance(instance);
            }
            // Analysis by increments
            else {
                int[] indices = windowIndices(this.windowSize);
                // Remove oldest chunk in tail model (tail chunk)
                this.previousModel.removeInstances(indices);

                // Move middle chunk from head to tail model
                Instances middleInstances = this.currentModel.removeInstances(indices);
                this.previousModel.addInstances(middleInstances);
                this.currentIndex += middleInstances.size();

                // Add new chunk to head model
                this.currentModel.addInstance(instance);
            }
        }
     }
}
