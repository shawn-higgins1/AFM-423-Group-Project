�	�v稿5@�v稿5@!�v稿5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�v稿5@A׾�^ 3@1�:��?AEGr��?I��h���@*effff�G@)       =2U
Iterator::Model::ParallelMapV2-C��6�?!
�<�;@)-C��6�?1
�<�;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�{�Pk�?!E]t�E;@)'�����?1����6@:Preprocessing2F
Iterator::Modela2U0*��?!�v-��KD@)-C��6z?1
�<�+@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate{�G�z�?!x[�C$5@)�HP�x?1��y���)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�q����o?!N�䁐} @)�q����o?1N�䁐} @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�?Ɯ?!x��s:�M@)ŏ1w-!o?1����Q @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!��)kʚ@)/n��b?1��)kʚ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�+e�X�?!��u��8@)Ǻ���V?1H
5λ�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�C��X@Q}/=�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	A׾�^ 3@A׾�^ 3@!A׾�^ 3@      ��!       "	�:��?�:��?!�:��?*      ��!       2	EGr��?EGr��?!EGr��?:	��h���@��h���@!��h���@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�C��X@y}/=�?�"5
sequential/dense/MatMulMatMulPB��UԦ?!PB��UԦ?0"C
%gradient_tape/sequential/dense/MatMulMatMul����J�?!������?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�!W�]�?!���,�?"7
sequential/dense_1/MatMulMatMul�!W�]�?!��6!@�?0"
Sum_7SumJ��[;L�?!�f��Ǩ�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam����J�?!X�*b&2�?"!
Adam/PowPow����J�?! �7���?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch����J�?!�|1�D�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad����J�?!tٴ�B��?"E
'gradient_tape/sequential/dense_1/MatMulMatMul:D�4ѓ?!����4$�?0Q      Y@Y��/Ċ�0@a�	�N]�T@q�c"�}X@y��l!��?"�
both�Your program is POTENTIALLY input-bound because 87.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 