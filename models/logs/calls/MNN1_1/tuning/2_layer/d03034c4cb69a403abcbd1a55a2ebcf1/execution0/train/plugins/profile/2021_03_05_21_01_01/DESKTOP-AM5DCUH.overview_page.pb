�	�;��7@�;��7@!�;��7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-�;��7@��Q,��4@1�`��p�?A�HP��?I E���D@*	������O@2U
Iterator::Model::ParallelMapV28��d�`�?!N��J?@)8��d�`�?1N��J?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�ݓ��Z�?!�R��=@)L7�A`�?1e�Cj��9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�������?!΀Pr�3@)�� �rh�?1��{mĺ*@:Preprocessing2F
Iterator::Model�#�����?!Dj��VvE@)���_vO~?1q��;E'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!<Eg@(@)����Mbp?1<Eg@(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip����镢?!��}��L@)F%u�k?1���!5�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!��sHM0@)a2U0*�c?1��sHM0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�!��u��?!�!5�x+6@)-C��6Z?1ʝ��3 @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 86.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�12.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIX�bM`�X@Q�A��g�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��Q,��4@��Q,��4@!��Q,��4@      ��!       "	�`��p�?�`��p�?!�`��p�?*      ��!       2	�HP��?�HP��?!�HP��?:	 E���D@ E���D@! E���D@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qX�bM`�X@y�A��g�?�"5
sequential/dense/MatMulMatMultּh�X�?!tּh�X�?0"C
%gradient_tape/sequential/dense/MatMulMatMul��>J�?!�^���5�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul��>J�?!R󜿻?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��>J�?!�"<	�$�?"7
sequential/dense_1/MatMulMatMul��>J�?!|�si�?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul��5�X�?!���߇��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad��j�X�?!�U\y�?�?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam��>J�?!��3��a�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch��>J�?!Χ�/B�?"7
sequential/dense_2/MatMulMatMul��>J�?!<fq�cS�?0Q      Y@Y>����/@aX�i��U@q�AѠhPX@y�2���?"�
both�Your program is POTENTIALLY input-bound because 86.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�12.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 